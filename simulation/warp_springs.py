# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Spring forces + semi-implicit Euler from NVIDIA/warp/warp/examples/benchmarks/benchmark_cloth_warp.py
# Gravity along -Z (Z up; sheet rests in the XY plane).

from __future__ import annotations

import math

import numpy as np
import warp as wp

from simulation.cloth_grid import ClothGrid


@wp.kernel
def eval_springs(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    f: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]
    xi = x[i]
    xj = x[j]
    vi = v[i]
    vj = v[j]
    xij = xi - xj
    vij = vi - vj
    l = wp.length(xij)
    l_inv = 1.0 / l
    dir = xij * l_inv
    c = l - rest
    dcdt = wp.dot(dir, vij)
    fs = dir * (ke * c + kd * dcdt)
    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def integrate_particles(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    w: wp.array[float],
    dt: float,
):
    tid = wp.tid()
    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]
    if inv_mass <= 0.0:
        v[tid] = wp.vec3()
        f[tid] = wp.vec3()
        return
    g = wp.vec3(0.0, 0.0, -9.81)
    v1 = v0 + (f0 * inv_mass + g) * dt
    x1 = x0 + v1 * dt
    x[tid] = x1
    v[tid] = v1
    f[tid] = wp.vec3()


class WarpSpringIntegrator:
    """Mass-spring sheet on CUDA (or CPU) via Warp."""

    def __init__(self, cloth: ClothGrid, device: str = "cuda:0") -> None:
        self.device = wp.get_device(device)
        self.cloth = cloth
        with wp.ScopedDevice(self.device):
            self.positions = wp.from_numpy(cloth.positions, dtype=wp.vec3)
            self.positions_host = wp.from_numpy(cloth.positions, dtype=wp.vec3, device="cpu")
            self.invmass = wp.from_numpy(cloth.inv_masses, dtype=float)
            self.velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.forces = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.spring_indices = wp.from_numpy(cloth.spring_indices, dtype=int)
            self.spring_lengths = wp.from_numpy(cloth.spring_lengths, dtype=float)
            self.spring_stiffness = wp.from_numpy(cloth.spring_stiffness, dtype=float)
            self.spring_damping = wp.from_numpy(cloth.spring_damping, dtype=float)

    def simulate(self, dt: float, substeps: int) -> np.ndarray:
        sim_dt = dt / float(max(1, substeps))
        for _ in range(substeps):
            wp.launch(
                kernel=eval_springs,
                dim=self.cloth.num_springs,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.spring_indices,
                    self.spring_lengths,
                    self.spring_stiffness,
                    self.spring_damping,
                    self.forces,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=integrate_particles,
                dim=self.cloth.num_particles,
                inputs=[self.positions, self.velocities, self.forces, self.invmass, sim_dt],
                device=self.device,
            )
        if self.device.is_cuda:
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()
            return self.positions_host.numpy()
        return self.positions.numpy()

    def numpy_positions(self) -> np.ndarray:
        if self.device.is_cuda:
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()
            return self.positions_host.numpy()
        return self.positions.numpy()

    def set_host_positions(self, xyz: np.ndarray) -> None:
        """Upload (N,3) float32 positions from NumPy."""
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        h = wp.from_numpy(xyz, dtype=wp.vec3, device="cpu")
        with wp.ScopedDevice(self.device):
            wp.copy(self.positions, h)

    def add_velocity_impulse(self, center_u: int, center_v: int, radius: float, strength: float) -> None:
        """Kick interior vertices near (center_u, center_v) with +Z velocity impulse."""
        v_host = wp.zeros(self.cloth.num_particles, dtype=wp.vec3, device="cpu")
        with wp.ScopedDevice(self.device):
            wp.copy(v_host, self.velocities)
        wp.synchronize()
        vel = v_host.numpy().reshape(-1, 3).copy()
        nu, nv = self.cloth.nu, self.cloth.nv
        r2 = radius * radius
        for v in range(nv):
            for u in range(nu):
                du = float(u - center_u)
                dv = float(v - center_v)
                if du * du + dv * dv > r2:
                    continue
                i = v * nu + u
                if self.cloth.inv_masses[i] <= 0.0:
                    continue
                vel[i, 2] += strength
        with wp.ScopedDevice(self.device):
            wp.copy(self.velocities, wp.from_numpy(vel.astype(np.float32), dtype=wp.vec3, device="cpu"))

    def add_velocity_impulses_uv_gaussian(
        self,
        centers_uv: np.ndarray,
        strength: float,
        radius_uv: float,
        nu: int,
        nv: int,
    ) -> None:
        """
        Add +Z velocity to vertices near each (u, v) center in grid-index space.
        Gaussian falloff in UV distance (same units as integer grid steps).
        """
        centers_uv = np.asarray(centers_uv, dtype=np.float64).reshape(-1, 2)
        if centers_uv.shape[0] == 0:
            return
        sigma = max(float(radius_uv) * 0.5, 1e-6)
        sigma2 = 2.0 * sigma * sigma
        st = float(strength)

        v_host = wp.zeros(self.cloth.num_particles, dtype=wp.vec3, device="cpu")
        with wp.ScopedDevice(self.device):
            wp.copy(v_host, self.velocities)
        wp.synchronize()
        vel = v_host.numpy().reshape(-1, 3).copy()

        for ci in range(centers_uv.shape[0]):
            uc = float(centers_uv[ci, 0])
            vc = float(centers_uv[ci, 1])
            du0 = int(max(0, math.floor(uc - 4.0 * sigma)))
            du1 = int(min(nu - 1, math.ceil(uc + 4.0 * sigma)))
            dv0 = int(max(0, math.floor(vc - 4.0 * sigma)))
            dv1 = int(min(nv - 1, math.ceil(vc + 4.0 * sigma)))
            for vv in range(dv0, dv1 + 1):
                for uu in range(du0, du1 + 1):
                    i = vv * nu + uu
                    if self.cloth.inv_masses[i] <= 0.0:
                        continue
                    du = float(uu) - uc
                    dv = float(vv) - vc
                    d2 = du * du + dv * dv
                    w = math.exp(-d2 / sigma2)
                    if w < 1e-8:
                        continue
                    vel[i, 2] += st * w

        with wp.ScopedDevice(self.device):
            wp.copy(self.velocities, wp.from_numpy(vel.astype(np.float32), dtype=wp.vec3, device="cpu"))
