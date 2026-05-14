"""
Multi-agent flow on a deformed cloth grid: p5-style damped 2D noise angle field (fixed for a run),
UV-parameterized agents on the surface with periodic UV reseed,
and batched Z velocity impulses to the sheet (Warp integrator).

Agents that reach the **clamped sheet boundary** in UV are marked inactive until the next reseed:
no advection, no impulse. Impulse direction can blend from world ±Z toward the **local deformed mesh normal** as the sheet
gains out-of-plane motion (see ``compute_agent_impulse_directions``).

Each reseed assigns a random **+Z or −Z** impulse direction per agent (fixed until the next reseed).
Impulse UV radius can **swell** in high–flow-curvature regions (spatial variation of the angle field),
by up to a configurable fraction of the per-agent base radius (default +10%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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


def bilinear_mesh_normal_world(
    positions: np.ndarray,
    nu: int,
    nv: int,
    uf: float,
    vf: float,
    *,
    duv_eps: float = 0.35,
) -> np.ndarray:
    """
    Unit world-space normal of the deformed bilinear sheet at float UV (u, v), from central differences.
    Orientation is chosen so ``n_z >= 0`` (outward "up" when the patch faces +Z).
    """
    e = float(max(duv_eps, 1e-4))
    pu = bilinear_mesh_point(positions, nu, nv, uf + e, vf)
    pmd = bilinear_mesh_point(positions, nu, nv, uf - e, vf)
    pv = bilinear_mesh_point(positions, nu, nv, uf, vf + e)
    pvn = bilinear_mesh_point(positions, nu, nv, uf, vf - e)
    du = (pu - pmd) / (2.0 * e)
    dv = (pv - pvn) / (2.0 * e)
    c = np.cross(du, dv)
    ln = float(np.linalg.norm(c))
    if ln < 1e-18:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = (c / ln).astype(np.float64)
    if float(n[2]) < 0.0:
        n = -n
    return np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)


def compute_agent_impulse_directions(
    positions: np.ndarray,
    nu: int,
    nv: int,
    uv: np.ndarray,
    z_sign: np.ndarray,
    magnitude: np.ndarray,
    *,
    tilt_max: float,
    deform_rms_z: float,
    deform_rms_ref: float,
) -> np.ndarray:
    """
    Per-agent unit impulse direction in world space for kicks with **non-negative** scalar strength.

    ``z_sign`` is ±1 per agent (from ``MeshFlowAgentSwarm.impulse_z_sign``). ``magnitude`` is the
    peak speed scale per agent (zero = inactive).

    Blends world ±Z with the local mesh normal at the agent UV. The blend fraction is
    ``tilt_max * min(1, rms_z / ref)`` using global RMS Z of the mesh so tilting **starts** as the
    sheet deforms, up to ``tilt_max`` (e.g. 0.2 → at most ~20% toward the normal, relative to ±Z).
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    magv = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    sgv = np.asarray(z_sign, dtype=np.float64).reshape(-1)
    n_ag = int(uv.shape[0])
    if magv.shape[0] != n_ag or sgv.shape[0] != n_ag:
        raise ValueError("magnitude and z_sign length must match uv rows")
    out = np.zeros((n_ag, 3), dtype=np.float64)
    out[:, 2] = 1.0
    ref = max(float(deform_rms_ref), 1e-12)
    f = float(max(0.0, tilt_max)) * min(1.0, float(deform_rms_z) / ref)

    for i in range(n_ag):
        mag = float(magv[i])
        if mag <= 0.0:
            out[i, :] = (0.0, 0.0, 1.0)
            continue
        s = 1.0 if float(sgv[i]) >= 0.0 else -1.0
        ez = np.array([0.0, 0.0, s], dtype=np.float64)
        if f <= 1e-15:
            out[i, :] = ez
            continue
        nm = bilinear_mesh_normal_world(positions, nu, nv, float(uv[i, 0]), float(uv[i, 1]))
        raw = (1.0 - f) * ez + f * s * nm
        rn = float(np.linalg.norm(raw))
        if rn < 1e-15:
            out[i, :] = ez
        else:
            out[i, :] = raw / rn
    return out


def discrete_laplacian_position_mag(positions: np.ndarray, nu: int, nv: int, uf: float, vf: float) -> float:
    """
    Magnitude of the 5-point discrete Laplacian of deformed positions at the grid cell nearest (uf, vf).
    Larger values indicate stronger local bending / curvature relative to neighbors.
    """
    if nu < 3 or nv < 3:
        return 0.0
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    uc = int(np.clip(round(float(uf)), 1, nu - 2))
    vc = int(np.clip(round(float(vf)), 1, nv - 2))

    def idx(u: int, v: int) -> int:
        return v * nu + u

    i0 = idx(uc, vc)
    lap = p[i0] - 0.25 * (
        p[idx(uc - 1, vc)]
        + p[idx(uc + 1, vc)]
        + p[idx(uc, vc - 1)]
        + p[idx(uc, vc + 1)]
    )
    return float(np.linalg.norm(lap))


def compute_agent_impulse_strengths(
    positions: np.ndarray,
    nu: int,
    nv: int,
    uv: np.ndarray,
    alive: np.ndarray,
    base_strength: float,
    *,
    curvature_grade_frac: float = 0.2,
) -> np.ndarray:
    """
    Per-agent peak impulse strength. Dead agents get 0.

    If ``curvature_grade_frac`` > 0, alive agents are scaled by relative Laplacian magnitude among
    alive agents this frame: ``base * (1 - f + 2*f*t)`` with ``t`` in [0,1], hence in
    ``[base*(1-f), base*(1+f)]`` (e.g. f=0.2 → ±20%). If f <= 0, all alive agents use ``base_strength``.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = int(uv.shape[0])
    alive_b = np.asarray(alive, dtype=bool).reshape(-1)
    if alive_b.shape[0] != n:
        raise ValueError(f"alive length {alive_b.shape[0]} must match uv rows {n}")
    out = np.zeros(n, dtype=np.float64)
    bs = float(base_strength)
    if bs <= 0.0:
        return out
    f = float(max(0.0, curvature_grade_frac))
    out[~alive_b] = 0.0
    idx_alive = np.flatnonzero(alive_b)
    if idx_alive.size == 0:
        return out
    if f <= 0.0:
        out[alive_b] = bs
        return out
    if nu < 3 or nv < 3:
        out[alive_b] = bs
        return out
    kappa = np.zeros(n, dtype=np.float64)
    for i in idx_alive:
        kappa[i] = discrete_laplacian_position_mag(positions, nu, nv, float(uv[i, 0]), float(uv[i, 1]))
    ka = kappa[alive_b]
    kmin = float(ka.min())
    kmax = float(ka.max())
    if kmax - kmin < 1e-18:
        out[alive_b] = bs
        return out
    t = np.zeros(n, dtype=np.float64)
    t[alive_b] = (kappa[alive_b] - kmin) / (kmax - kmin)
    t = np.clip(t, 0.0, 1.0)
    mult = (1.0 - f) + 2.0 * f * t
    out = bs * mult * alive_b.astype(np.float64)
    return out


def flow_angle_curvature_proxy(flow: P5StyleFlowField, wx: float, wy: float) -> float:
    """
    Magnitude of the world-space gradient of ``flow.angle_at`` (central differences, angle-wrapped).
    Large values indicate rapid turning of the flow direction (high "curvature" of the field).
    """
    p = flow.params
    hw = max(float(p.plane_width) * 0.5, 1e-9)
    hd = max(float(p.plane_depth) * 0.5, 1e-9)
    e = 0.025 * min(hw, hd)
    e = max(float(e), 1e-6)
    ac = flow.angle_at(wx, wy)
    if not math.isfinite(ac):
        return 0.0

    def rel(dx: float, dy: float) -> float:
        aa = flow.angle_at(wx + dx, wy + dy)
        if not math.isfinite(aa):
            return 0.0
        return math.atan2(math.sin(aa - ac), math.cos(aa - ac))

    gx = (rel(e, 0.0) - rel(-e, 0.0)) / (2.0 * e)
    gy = (rel(0.0, e) - rel(0.0, -e)) / (2.0 * e)
    return float(math.hypot(gx, gy))


def compute_agent_impulse_radii_flow_swollen(
    flow: P5StyleFlowField,
    positions: np.ndarray,
    nu: int,
    nv: int,
    uv: np.ndarray,
    alive: np.ndarray,
    base_radii_uv: np.ndarray,
    *,
    swell_frac: float = 0.1,
) -> np.ndarray:
    """
    Per-agent UV impulse radius = ``base_radii * (1 + swell_frac * t)`` for alive agents, where ``t``
    in [0, 1] ranks flow-angle spatial curvature (``flow_angle_curvature_proxy``) at the agent's
    deformed surface point among alive agents this frame. Dead agents keep ``base_radii``.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = int(uv.shape[0])
    base = np.asarray(base_radii_uv, dtype=np.float64).reshape(-1)
    if base.shape[0] != n:
        raise ValueError(f"base_radii_uv length {base.shape[0]} must match uv rows {n}")
    out = base.copy()
    sf = float(max(0.0, swell_frac))
    if sf <= 0.0:
        return out
    alive_b = np.asarray(alive, dtype=bool).reshape(-1)
    if alive_b.shape[0] != n:
        raise ValueError(f"alive length {alive_b.shape[0]} must match uv rows {n}")
    idx_alive = np.flatnonzero(alive_b)
    if idx_alive.size == 0:
        return out
    kappa = np.zeros(n, dtype=np.float64)
    for i in idx_alive:
        pt = bilinear_mesh_point(positions, nu, nv, float(uv[i, 0]), float(uv[i, 1]))
        kappa[i] = flow_angle_curvature_proxy(flow, float(pt[0]), float(pt[1]))
    ka = kappa[alive_b]
    kmin = float(ka.min())
    kmax = float(ka.max())
    if kmax - kmin < 1e-18:
        return out
    t = np.zeros(n, dtype=np.float64)
    t[alive_b] = (kappa[alive_b] - kmin) / (kmax - kmin)
    t = np.clip(t, 0.0, 1.0)
    mult = np.ones(n, dtype=np.float64)
    mult[alive_b] = 1.0 + sf * t[alive_b]
    return base * mult


@dataclass
class FlowFieldParams:
    """
    Flow angles from smooth value-noise on a torus, sampled in **sheet-normalized** XY:
    ``sx = x / (plane_width/2)``, ``sy = y / (plane_depth/2)`` so the field matches cloth extent.

    ``damp_x`` / ``damp_y`` are dimensionless smoothness (larger ⇒ smoother, slower angle change
    across the sheet; smaller ⇒ more dynamic / twisty trajectories). Typical range ~0.15–0.7.
    """

    plane_width: float = 24.0
    plane_depth: float = 24.0
    damp_x: float = 0.38
    damp_y: float = 0.38
    spatial_scale: float = 3.6
    simplex_style: bool = False
    grid_n: int = 40


class P5StyleFlowField:
    """
    Smooth value-noise on a torus; angle = noise * pi (180°) or noise * pi/2 (90°) like p5.
    Samples noise using **normalized** surface coordinates so the flow matches sheet scale.
    """

    def __init__(self, params: FlowFieldParams | None = None) -> None:
        self.params = params or FlowFieldParams()
        self._vals: np.ndarray | None = None
        self._phase_x = 0.0
        self._phase_y = 0.0

    def regenerate(self, rng: np.random.Generator) -> None:
        gn = max(4, int(self.params.grid_n))
        self._vals = rng.random((gn, gn), dtype=np.float64)
        self._phase_x = float(rng.uniform(0.0, 200.0))
        self._phase_y = float(rng.uniform(0.0, 200.0))

    def angle_at(self, wx: float, wy: float) -> float:
        if self._vals is None or not (math.isfinite(wx) and math.isfinite(wy)):
            return 0.0
        p = self.params
        hw = max(float(p.plane_width) * 0.5, 1e-9)
        hd = max(float(p.plane_depth) * 0.5, 1e-9)
        sx = wx / hw
        sy = wy / hd
        sc = max(float(p.spatial_scale), 1e-6)
        fx = (sx / max(p.damp_x, 1e-9) + self._phase_x) * sc
        fy = (sy / max(p.damp_y, 1e-9) + self._phase_y) * sc
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

    Optional **edge repulsion** pushes agents away from the sheet boundary in UV so they
    do not pool along the rim (in addition to ``margin_cells`` clamp near anchored edges).

    After each step, agents whose UV sits on the clamped boundary are marked **inactive**
    until ``reseed``: they no longer move or contribute impulses.

    On each ``reseed``, ``impulse_z_sign`` is drawn ±1 per agent (impulse along +Z or −Z until next reseed).
    """

    def __init__(
        self,
        n_agents: int,
        nu: int,
        nv: int,
        *,
        margin_cells: float = 1.5,
        edge_keepout_cells: float = 4.0,
        edge_repel_strength: float = 0.22,
        impulse_radius_base: float = 2.5,
        impulse_radius_jitter_frac: float = 0.1,
    ) -> None:
        self.n_agents = int(n_agents)
        self.nu = int(nu)
        self.nv = int(nv)
        self.margin = float(margin_cells)
        self.edge_keepout = float(max(0.0, edge_keepout_cells))
        self.edge_repel = float(max(0.0, edge_repel_strength))
        self.impulse_radius_base = float(max(impulse_radius_base, 1e-9))
        self.impulse_radius_jitter_frac = float(max(0.0, impulse_radius_jitter_frac))
        self.uv = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.impulse_radius_uv = np.full(self.n_agents, self.impulse_radius_base, dtype=np.float64)
        self.agent_alive = np.ones(self.n_agents, dtype=bool)
        self.impulse_z_sign = np.ones(self.n_agents, dtype=np.float64)

    def _refresh_impulse_radii(self, rng: np.random.Generator) -> None:
        """Per-agent UV impulse radius; uniform jitter in [base*(1-f), base*(1+f)] on each reseed."""
        f = self.impulse_radius_jitter_frac
        b = self.impulse_radius_base
        if f <= 0.0:
            self.impulse_radius_uv[:] = b
            return
        lo = b * (1.0 - f)
        hi = b * (1.0 + f)
        self.impulse_radius_uv[:] = rng.uniform(lo, hi, size=self.n_agents)

    def _refresh_impulse_z_sign(self, rng: np.random.Generator) -> None:
        """Random +1 / −1 per agent; impulse Z direction stays fixed until the next reseed."""
        self.impulse_z_sign[:] = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=self.n_agents)

    def _interior_pad_uv(self) -> tuple[float, float]:
        """Minimum distance from u=0 / v=0 edges for reseed (and repulsion band width)."""
        pad_u = max(self.margin, self.edge_keepout)
        pad_v = max(self.margin, self.edge_keepout)
        return pad_u, pad_v

    def _repel_from_sheet_edges(self) -> None:
        """Inward UV nudge when within ``edge_keepout`` of any sheet side (quadratic falloff)."""
        if self.edge_repel <= 0.0 or self.edge_keepout <= 0.0:
            return
        w = max(self.edge_keepout, 1e-6)
        s = self.edge_repel
        umax = float(self.nu) - 1.0
        vmax = float(self.nv) - 1.0
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            u = float(self.uv[i, 0])
            v = float(self.uv[i, 1])
            if u < w:
                t = (w - u) / w
                self.uv[i, 0] += s * (t * t)
            u = float(self.uv[i, 0])
            if (umax - u) < w:
                t = (umax - u) / w
                self.uv[i, 0] -= s * (t * t)
            v = float(self.uv[i, 1])
            if v < w:
                t = (w - v) / w
                self.uv[i, 1] += s * (t * t)
            v = float(self.uv[i, 1])
            if (vmax - v) < w:
                t = (vmax - v) / w
                self.uv[i, 1] -= s * (t * t)

    def reseed(self, rng: np.random.Generator) -> None:
        pad_u, pad_v = self._interior_pad_uv()
        lo_u = pad_u
        hi_u = max(lo_u + 1e-6, float(self.nu - 1) - pad_u)
        lo_v = pad_v
        hi_v = max(lo_v + 1e-6, float(self.nv - 1) - pad_v)
        self.uv[:, 0] = rng.uniform(lo_u, hi_u, size=self.n_agents)
        self.uv[:, 1] = rng.uniform(lo_v, hi_v, size=self.n_agents)
        self._refresh_impulse_radii(rng)
        self._refresh_impulse_z_sign(rng)
        self.agent_alive[:] = True

    def _mark_dead_at_uv_boundary(self) -> None:
        """Kill agents sitting on the clamped interior boundary (sheet edge in UV)."""
        tol = 1e-5
        u_lo = float(self.margin)
        u_hi = float(self.nu) - 1.0 - float(self.margin)
        v_lo = float(self.margin)
        v_hi = float(self.nv) - 1.0 - float(self.margin)
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            u = float(self.uv[i, 0])
            v = float(self.uv[i, 1])
            if u <= u_lo + tol or u >= u_hi - tol or v <= v_lo + tol or v >= v_hi - tol:
                self.agent_alive[i] = False

    def step(self, positions: np.ndarray, flow: P5StyleFlowField, step_cells: float) -> None:
        sc = float(step_cells)
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            uf, vf = self.uv[i]
            p = bilinear_mesh_point(positions, self.nu, self.nv, uf, vf)
            ang = flow.angle_at(float(p[0]), float(p[1]))
            if not math.isfinite(ang):
                ang = 0.0
            self.uv[i, 0] += math.cos(ang) * sc
            self.uv[i, 1] += math.sin(ang) * sc
        self._repel_from_sheet_edges()
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            self.uv[i, 0] = float(np.clip(self.uv[i, 0], self.margin, self.nu - 1.0 - self.margin))
            self.uv[i, 1] = float(np.clip(self.uv[i, 1], self.margin, self.nv - 1.0 - self.margin))
        self._mark_dead_at_uv_boundary()


def save_flow_field_viz_png(
    path: str | Path,
    flow: P5StyleFlowField,
    *,
    grid: int = 72,
    dpi: int = 140,
) -> None:
    """
    Save a quiver plot of ``flow.angle_at`` over normalized sheet coordinates
    ``x/(w/2), y/(d/2)`` in [-1, 1]² (matches agent sampling on the rest XY plane).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    p = flow.params
    hw = max(float(p.plane_width) * 0.5, 1e-9)
    hd = max(float(p.plane_depth) * 0.5, 1e-9)
    n = int(max(8, grid))
    su = np.linspace(-1.0, 1.0, n)
    sv = np.linspace(-1.0, 1.0, n)
    U, V = np.meshgrid(su, sv, indexing="xy")
    wx = U * hw
    wy = V * hd
    ang = np.vectorize(lambda x, y: float(flow.angle_at(float(x), float(y))))(wx, wy)

    mode = "simplex (90°)" if p.simplex_style else "full (180°)"
    fig, ax = plt.subplots(figsize=(7.5, 6.8), facecolor="#f0f0f2")
    ax.set_facecolor("#fafafa")
    step = max(1, n // 32)
    Qx = U[::step, ::step]
    Qy = V[::step, ::step]
    Ag = ang[::step, ::step]
    ax.quiver(
        Qx.ravel(),
        Qy.ravel(),
        np.cos(Ag).ravel(),
        np.sin(Ag).ravel(),
        angles="xy",
        scale_units="xy",
        scale=14.0,
        width=0.0032,
        color="0.18",
        headwidth=3.2,
        headlength=4.0,
    )
    ax.set_xlabel(r"$x\,/\,(w/2)$", color="0.2")
    ax.set_ylabel(r"$y\,/\,(d/2)$", color="0.2")
    ax.set_title(
        f"Agent flow (XY plane, Z up)  |  damp={p.damp_x:.3g}, spatial_scale={p.spatial_scale:.3g}, "
        f"noise {p.grid_n}×{p.grid_n}, angles: {mode}",
        fontsize=10,
        color="0.15",
    )
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.78", linewidth=0.65, zorder=0)
    ax.axvline(0.0, color="0.78", linewidth=0.65, zorder=0)
    ax.grid(True, alpha=0.38, color="0.55")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
