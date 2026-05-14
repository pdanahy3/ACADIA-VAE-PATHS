"""
Multi-agent flow on a deformed cloth grid: p5-style damped 2D noise angle field,
UV-parameterized agents on the surface, periodic reseed + field regeneration,
and batched Z velocity impulses to the sheet (Warp integrator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear_sample_grid(vals: np.ndarray, fx: float, fy: float) -> float:
    """vals shape (gh, gw), sample at fractional coords in [0, gw-1] x [0, gh-1] with wrap."""
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return 0.5
    gw, gh = vals.shape[1], vals.shape[0]
    if gw < 2 or gh < 2:
        return float(vals[0, 0])
    fx = fx % (gw - 1)
    fy = fy % (gh - 1)
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = min(x0 + 1, gw - 1)
    y1 = min(y0 + 1, gh - 1)
    tx = fx - x0
    ty = fy - y0
    tx = _smoothstep(tx)
    ty = _smoothstep(ty)
    v00 = float(vals[y0, x0])
    v10 = float(vals[y0, x1])
    v01 = float(vals[y1, x0])
    v11 = float(vals[y1, x1])
    a = v00 * (1 - tx) + v10 * tx
    b = v01 * (1 - tx) + v11 * tx
    return float(a * (1 - ty) + b * ty)


def bilinear_mesh_point(positions: np.ndarray, nu: int, nv: int, uf: float, vf: float) -> np.ndarray:
    """Interpolate deformed position at float grid indices (u along nu, v along nv)."""
    uf = float(np.clip(uf, 0.0, nu - 1))
    vf = float(np.clip(vf, 0.0, nv - 1))
    u0 = int(math.floor(uf))
    v0 = int(math.floor(vf))
    u1 = min(u0 + 1, nu - 1)
    v1 = min(v0 + 1, nv - 1)
    tu = uf - u0
    tv = vf - v0
    tu = float(_smoothstep(tu))
    tv = float(_smoothstep(tv))
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    i00 = v0 * nu + u0
    i10 = v0 * nu + u1
    i01 = v1 * nu + u0
    i11 = v1 * nu + u1
    a = p[i00] * (1 - tu) + p[i10] * tu
    b = p[i01] * (1 - tu) + p[i11] * tu
    out = (a * (1 - tv) + b * tv).astype(np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class FlowFieldParams:
    """Analogous to p5 noise / simplex2 angle scaling with x_damp, y_damp."""

    damp_x: float = 40.0
    damp_y: float = 40.0
    simplex_style: bool = False
    grid_n: int = 32


class P5StyleFlowField:
    """
    Smooth value-noise on a torus; angle = noise * pi (180°) or noise * pi/2 (90°) like p5.
    Sampling uses world (x, y) scaled by 1/damp_* plus random phase per regenerate().
    """

    def __init__(self, params: FlowFieldParams | None = None) -> None:
        self.params = params or FlowFieldParams()
        self._vals: np.ndarray | None = None
        self._phase_x = 0.0
        self._phase_y = 0.0
        self._scale = 2.8

    def regenerate(self, rng: np.random.Generator) -> None:
        gn = max(4, int(self.params.grid_n))
        self._vals = rng.random((gn, gn), dtype=np.float64)
        self._phase_x = float(rng.uniform(0.0, 200.0))
        self._phase_y = float(rng.uniform(0.0, 200.0))

    def angle_at(self, wx: float, wy: float) -> float:
        if self._vals is None or not (math.isfinite(wx) and math.isfinite(wy)):
            return 0.0
        p = self.params
        fx = (wx / max(p.damp_x, 1e-6) + self._phase_x) * self._scale
        fy = (wy / max(p.damp_y, 1e-6) + self._phase_y) * self._scale
        t = _bilinear_sample_grid(self._vals, fx, fy)
        if not math.isfinite(t):
            return 0.0
        if p.simplex_style:
            return t * (0.5 * math.pi)
        return t * math.pi


class MeshFlowAgentSwarm:
    """
    Agents live at float (u, v) in grid index space; each step reads flow angle from
    world (x, y) of the bilinear surface point and advects in (u, v) by the same angle
    (isotropic step in index space — reasonable when dx ~ dy).
    """

    def __init__(
        self,
        n_agents: int,
        nu: int,
        nv: int,
        *,
        margin_cells: float = 1.5,
    ) -> None:
        self.n_agents = int(n_agents)
        self.nu = int(nu)
        self.nv = int(nv)
        self.margin = float(margin_cells)
        self.uv = np.zeros((self.n_agents, 2), dtype=np.float64)

    def reseed(self, rng: np.random.Generator) -> None:
        lo_u = self.margin
        hi_u = max(lo_u + 1e-6, self.nu - 1.0 - self.margin)
        lo_v = self.margin
        hi_v = max(lo_v + 1e-6, self.nv - 1.0 - self.margin)
        self.uv[:, 0] = rng.uniform(lo_u, hi_u, size=self.n_agents)
        self.uv[:, 1] = rng.uniform(lo_v, hi_v, size=self.n_agents)

    def step(self, positions: np.ndarray, flow: P5StyleFlowField, step_cells: float) -> None:
        sc = float(step_cells)
        for i in range(self.n_agents):
            uf, vf = self.uv[i]
            p = bilinear_mesh_point(positions, self.nu, self.nv, uf, vf)
            ang = flow.angle_at(float(p[0]), float(p[1]))
            if not math.isfinite(ang):
                ang = 0.0
            self.uv[i, 0] += math.cos(ang) * sc
            self.uv[i, 1] += math.sin(ang) * sc
            self.uv[i, 0] = float(np.clip(self.uv[i, 0], self.margin, self.nu - 1.0 - self.margin))
            self.uv[i, 1] = float(np.clip(self.uv[i, 1], self.margin, self.nv - 1.0 - self.margin))
