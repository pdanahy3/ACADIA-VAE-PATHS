# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Spring forces + semi-implicit Euler from NVIDIA/warp/warp/examples/benchmarks/benchmark_cloth_warp.py
# Gravity along -Z by default (Z up; sheet rests in the XY plane). Optional elastoplastic spring rest updates.

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
    gx: float,
    gy: float,
    gz: float,
    velocity_drag: float,
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
    g = wp.vec3(gx, gy, gz)
    v1 = v0 + (f0 * inv_mass + g) * dt
    if velocity_drag > 0.0:
        damp = wp.exp(-velocity_drag * dt)
        v1 = v1 * damp
    x1 = x0 + v1 * dt
    x[tid] = x1
    v[tid] = v1
    f[tid] = wp.vec3()


@wp.kernel
def apply_spring_plasticity(
    x: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_init_lengths: wp.array[float],
    yield_strain: float,
    plastic_creep: float,
    max_extra_strain: float,
):
    """Elastoplastic rest length: when |L - L0| > yield_strain * L0i, creep L0 toward L."""
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    L0 = spring_rest_lengths[tid]
    L0i = spring_init_lengths[tid]
    if L0i < 1.0e-12:
        return
    xi = x[i]
    xj = x[j]
    xij = xi - xj
    L = wp.length(xij)
    if L < 1.0e-12:
        return
    band = yield_strain * L0i
    de = L - L0
    new_L0 = L0
    if de > band:
        excess = de - band
        new_L0 = L0 + plastic_creep * excess
    elif de < -band:
        excess = de + band
        new_L0 = L0 + plastic_creep * excess
    lo = L0i * (1.0 - max_extra_strain)
    hi = L0i * (1.0 + max_extra_strain)
    spring_rest_lengths[tid] = wp.clamp(new_L0, lo, hi)


class WarpSpringIntegrator:
    """Mass-spring sheet on CUDA (or CPU) via Warp."""

    def __init__(
        self,
        cloth: ClothGrid,
        device: str = "cuda:0",
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        velocity_drag: float = 0.0,
        plastic_yield_strain: float = 0.0,
        plastic_creep: float = 0.12,
        plastic_max_strain: float = 0.45,
    ) -> None:
        self.device = wp.get_device(device)
        self.cloth = cloth
        self._gx, self._gy, self._gz = (float(gravity[0]), float(gravity[1]), float(gravity[2]))
        self.velocity_drag = float(max(0.0, velocity_drag))
        self.plastic_yield_strain = float(max(0.0, plastic_yield_strain))
        self.plastic_creep = float(np.clip(plastic_creep, 0.0, 1.0))
        self.plastic_max_strain = float(max(1e-6, plastic_max_strain))
        with wp.ScopedDevice(self.device):
            self.positions = wp.from_numpy(cloth.positions, dtype=wp.vec3)
            self.positions_host = wp.from_numpy(cloth.positions, dtype=wp.vec3, device="cpu")
            self.invmass = wp.from_numpy(cloth.inv_masses, dtype=float)
            self.velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.forces = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.spring_indices = wp.from_numpy(cloth.spring_indices, dtype=int)
            self.spring_lengths = wp.from_numpy(cloth.spring_lengths, dtype=float)
            sl_np = np.asarray(cloth.spring_lengths, dtype=np.float32)
            self.spring_init_lengths = wp.from_numpy(sl_np.copy(), dtype=float)
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
                inputs=[
                    self.positions,
                    self.velocities,
                    self.forces,
                    self.invmass,
                    float(self._gx),
                    float(self._gy),
                    float(self._gz),
                    float(self.velocity_drag),
                    sim_dt,
                ],
                device=self.device,
            )
        if self.plastic_yield_strain > 0.0 and self.plastic_creep > 0.0:
            wp.launch(
                kernel=apply_spring_plasticity,
                dim=self.cloth.num_springs,
                inputs=[
                    self.positions,
                    self.spring_indices,
                    self.spring_lengths,
                    self.spring_init_lengths,
                    float(self.plastic_yield_strain),
                    float(self.plastic_creep),
                    float(self.plastic_max_strain),
                ],
                device=self.device,
            )
        if self.device.is_cuda:
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()
            return self.positions_host.numpy()
        return self.positions.numpy()

    def numpy_spring_rest_lengths(self) -> np.ndarray:
        """Host copy of per-spring rest lengths (includes plastic drift)."""
        h = wp.zeros(self.cloth.num_springs, dtype=float, device="cpu")
        with wp.ScopedDevice(self.device):
            wp.copy(h, self.spring_lengths)
        wp.synchronize()
        return h.numpy().astype(np.float64).copy()

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
        strength: float | np.ndarray,
        radius_uv: float | np.ndarray,
        nu: int,
        nv: int,
        *,
        sigma_frac: float = 0.22,
        impulse_dir_world: np.ndarray | None = None,
    ) -> None:
        """
        Add velocity in a compact UV disk: no influence past ``radius_uv``; within the disk,
        weight falls off as a Gaussian with std ``radius_uv * sigma_frac`` (smaller ``sigma_frac``
        = sharper bump at the agent).

        ``radius_uv`` may be a scalar (same for every center) or a 1D array of length ``K`` matching
        ``centers_uv`` for per-agent radii.

        ``strength`` may be a scalar or a 1D array of length ``K`` for per-agent peak kick **magnitudes**
        along ``impulse_dir_world`` (use 0 for inactive agents). With default ``impulse_dir_world`` (+Z),
        **negative** values give −Z kicks. With a custom ``impulse_dir_world``, use **non-negative**
        magnitudes and bake ±Z into each direction row (e.g. agent ``impulse_z_sign``).

        ``impulse_dir_world`` optional shape ``(K, 3)``: unit-ish world direction per agent (normalized
        internally). If omitted, defaults to +Z ``(0, 0, 1)`` for every agent (pure world-Z kicks).

        Overlapping agents **do not stack**: per vertex the applied Δ**v** is the vector contribution
        from the agent with the largest ``‖strength * weight * direction‖``.
        """
        centers_uv = np.asarray(centers_uv, dtype=np.float64).reshape(-1, 2)
        k = int(centers_uv.shape[0])
        if k == 0:
            return
        radii = np.asarray(radius_uv, dtype=np.float64).reshape(-1)
        if radii.size == 1:
            radii = np.full(k, float(radii[0]), dtype=np.float64)
        elif radii.size != k:
            raise ValueError(
                f"radius_uv must be scalar or length {k} (centers), got shape {radii.shape}"
            )

        strengths = np.asarray(strength, dtype=np.float64).reshape(-1)
        if strengths.size == 1:
            st_per = np.full(k, float(strengths[0]), dtype=np.float64)
        elif strengths.size != k:
            raise ValueError(
                f"strength must be scalar or length {k} (centers), got shape {strengths.shape}"
            )
        else:
            st_per = strengths.astype(np.float64, copy=False)

        if impulse_dir_world is None:
            dir_rows = np.zeros((k, 3), dtype=np.float64)
            dir_rows[:, 2] = 1.0
        else:
            dir_rows = np.asarray(impulse_dir_world, dtype=np.float64).reshape(k, 3)
            if dir_rows.shape[0] != k:
                raise ValueError(
                    f"impulse_dir_world must have shape ({k}, 3), got {dir_rows.shape}"
                )
            ln = np.linalg.norm(dir_rows, axis=1, keepdims=True)
            ln = np.maximum(ln, 1e-12)
            dir_rows = dir_rows / ln

        sf = max(float(sigma_frac), 1e-4)

        v_host = wp.zeros(self.cloth.num_particles, dtype=wp.vec3, device="cpu")
        with wp.ScopedDevice(self.device):
            wp.copy(v_host, self.velocities)
        wp.synchronize()
        vel = v_host.numpy().reshape(-1, 3).copy()

        npt = nu * nv
        dv_pick_abs = np.zeros(npt, dtype=np.float64)
        dv_pick_val = np.zeros((npt, 3), dtype=np.float64)
        invm = np.asarray(self.cloth.inv_masses, dtype=np.float64).reshape(-1)

        for ci in range(k):
            st = float(st_per[ci])
            if st == 0.0:
                continue
            dxc, dyc, dzc = float(dir_rows[ci, 0]), float(dir_rows[ci, 1]), float(dir_rows[ci, 2])
            hard_r = max(float(radii[ci]), 1e-6)
            hard_r2 = hard_r * hard_r
            sigma = max(hard_r * sf, 1e-6)
            sigma2 = 2.0 * sigma * sigma
            uc = float(centers_uv[ci, 0])
            vc = float(centers_uv[ci, 1])
            du0 = int(max(0, math.floor(uc - hard_r - 1.0)))
            du1 = int(min(nu - 1, math.ceil(uc + hard_r + 1.0)))
            dv0 = int(max(0, math.floor(vc - hard_r - 1.0)))
            dv1 = int(min(nv - 1, math.ceil(vc + hard_r + 1.0)))
            for vv in range(dv0, dv1 + 1):
                for uu in range(du0, du1 + 1):
                    i = vv * nu + uu
                    if invm[i] <= 0.0:
                        continue
                    du = float(uu) - uc
                    dv = float(vv) - vc
                    d2 = du * du + dv * dv
                    if d2 > hard_r2:
                        continue
                    w = math.exp(-d2 / sigma2)
                    if w < 1e-10:
                        continue
                    sw = st * w
                    vx = sw * dxc
                    vy = sw * dyc
                    vz = sw * dzc
                    ca = math.sqrt(vx * vx + vy * vy + vz * vz)
                    if ca > dv_pick_abs[i]:
                        dv_pick_abs[i] = ca
                        dv_pick_val[i, 0] = vx
                        dv_pick_val[i, 1] = vy
                        dv_pick_val[i, 2] = vz

        vel += dv_pick_val.astype(np.float32)

        with wp.ScopedDevice(self.device):
            wp.copy(self.velocities, wp.from_numpy(vel.astype(np.float32), dtype=wp.vec3, device="cpu"))
