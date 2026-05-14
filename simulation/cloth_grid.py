# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from NVIDIA/warp/warp/examples/benchmarks/benchmark_cloth.py — grid topology,
# spring graph, and masses adapted for a rectangular sheet with anchored u-min / u-max edges
# (similar to scripts/sheet-metal-simulation.js).

from __future__ import annotations

import numpy as np


class ClothGrid:
    """
    Regular grid in the **XY** plane at **Z = 0** (horizontal sheet), **Z up** for out-of-plane motion,
    plus stretch / bend / shear springs (NVIDIA benchmark pattern).

    When ``anchor_u_edges`` is true, u=0 and u=nu-1 vertices use ``anchor_strength`` to scale effective
    mass (larger ⇒ firmer hold; values ≥ 1e10 are kinematic / fully fixed, matching the original benchmark).
    """

    def __init__(
        self,
        nu: int,
        nv: int,
        *,
        plane_width: float = 24.0,
        plane_depth: float = 24.0,
        stretch_stiffness: float = 1000.0,
        bend_stiffness: float = 1000.0,
        shear_stiffness: float = 1000.0,
        mass: float = 0.1,
        spring_damping: float = 10.0,
        anchor_u_edges: bool = True,
        anchor_strength: float = 1e12,
    ) -> None:
        self.nu = nu
        self.nv = nv
        self.plane_width = float(plane_width)
        self.plane_depth = float(plane_depth)
        spring_damp = float(spring_damping)
        mass = float(max(mass, 1e-12))
        inv_interior = 1.0 / mass
        # Edge inverse-mass = inv_interior / anchor_strength (heavier / "firmer" when strength is large).
        # Very large strength uses kinematic verts (inv_mass = 0) like the original benchmark.
        _RIGID_ANCHOR_STRENGTH = 1e10
        as_ = float(max(anchor_strength, 1.0))
        dx = self.plane_width / max(1, nu - 1)
        dy = self.plane_depth / max(1, nv - 1)
        lower = np.array(
            (-0.5 * self.plane_width, -0.5 * self.plane_depth, 0.0),
            dtype=np.float64,
        )
        self._lower = lower.astype(np.float32)
        self._dx = float(dx)
        self._dy = float(dy)

        self.triangles: list[int] = []
        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.inv_masses: list[float] = []
        self.spring_indices: list[int] = []
        self.spring_lengths: list[float] = []
        self.spring_stiffness: list[float] = []
        self.spring_damping: list[float] = []

        def idx(u: int, v: int) -> int:
            return v * nu + u

        def create_spring(i: int, j: int, stiffness: float) -> None:
            length = float(np.linalg.norm(np.asarray(self.positions[i]) - np.asarray(self.positions[j])))
            self.spring_indices.extend([i, j])
            self.spring_lengths.append(length)
            self.spring_stiffness.append(stiffness)
            self.spring_damping.append(spring_damp)

        for v in range(nv):
            for u in range(nu):
                p = lower + np.array((u * dx, v * dy, 0.0), dtype=np.float64)
                self.positions.append(p)
                self.velocities.append(np.zeros(3, dtype=np.float64))

                if u > 0 and v > 0:
                    self.triangles.extend(
                        [
                            idx(u - 1, v - 1),
                            idx(u, v - 1),
                            idx(u, v),
                            idx(u - 1, v - 1),
                            idx(u, v),
                            idx(u - 1, v),
                        ]
                    )

                if anchor_u_edges and (u == 0 or u == nu - 1):
                    if as_ >= _RIGID_ANCHOR_STRENGTH:
                        w = 0.0
                    else:
                        w = float(inv_interior / as_)
                else:
                    w = float(inv_interior)
                self.inv_masses.append(w)

        # structural springs (same topology as NVIDIA cloth benchmark)
        for v in range(nv):
            for u in range(nu):
                index0 = idx(u, v)
                if u > 0:
                    create_spring(index0, idx(u - 1, v), stretch_stiffness)
                if u > 1 and bend_stiffness > 0.0:
                    create_spring(index0, idx(u - 2, v), bend_stiffness)
                if v > 0 and u < nu - 1 and shear_stiffness > 0.0:
                    create_spring(index0, idx(u + 1, v - 1), shear_stiffness)
                if v > 0 and u > 0 and shear_stiffness > 0.0:
                    create_spring(index0, idx(u - 1, v - 1), shear_stiffness)

        for u in range(nu):
            for v in range(nv):
                index0 = idx(u, v)
                if v > 0:
                    create_spring(index0, idx(u, v - 1), stretch_stiffness)
                if v > 1 and bend_stiffness > 0.0:
                    create_spring(index0, idx(u, v - 2), bend_stiffness)

        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.velocities = np.asarray(self.velocities, dtype=np.float32)
        self.inv_masses = np.asarray(self.inv_masses, dtype=np.float32)
        self.spring_lengths = np.asarray(self.spring_lengths, dtype=np.float32)
        self.spring_indices = np.asarray(self.spring_indices, dtype=np.int32)
        self.spring_stiffness = np.asarray(self.spring_stiffness, dtype=np.float32)
        self.spring_damping = np.asarray(self.spring_damping, dtype=np.float32)

        self.num_particles = len(self.positions)
        self.num_springs = len(self.spring_lengths)
        self.num_tris = len(self.triangles) // 3

    def rest_positions(self) -> np.ndarray:
        """(N, 3) rest layout: XY plane at Z = 0 (Z up)."""
        nu, nv = self.nu, self.nv
        out = np.zeros((self.num_particles, 3), dtype=np.float32)
        i = 0
        for v in range(nv):
            for u in range(nu):
                out[i] = self._lower + np.array((u * self._dx, v * self._dy, 0.0), dtype=np.float32)
                i += 1
        return out
