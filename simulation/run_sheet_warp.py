#!/usr/bin/env python3
"""
CUDA sheet / cloth simulation using NVIDIA Warp (mass–spring grid).

Upstream reference (Apache-2.0):
  https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth.py
  https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth_warp.py

Run (from repo root):
  python -m simulation.run_sheet_warp --steps 300 --device cuda:0 --view3d --out-dir data/warp_run1

Requires NVIDIA Warp on PyPI as the package name warp-lang (do not install the unrelated PyPI package "warp"):

  python -m pip install warp-lang
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(_REPO))

try:
    import warp as wp
except ModuleNotFoundError:
    print(
        "Missing NVIDIA Warp. Install the PyPI package 'warp-lang' (import name is still 'warp'):\n"
        "  python -m pip install warp-lang\n"
        "Do not use: pip install warp   (that is a different, unrelated project).\n"
        "Use the same interpreter you run this script with (e.g. conda env ACADIA2026).",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from simulation.cloth_grid import ClothGrid
from simulation.displacement_png import fixed_domain_half_extent_from_rest, save_displacement_png
from simulation.mesh_flow_agents import (
    FlowFieldParams,
    MeshFlowAgentSwarm,
    P5StyleFlowField,
    compute_agent_impulse_directions,
    compute_agent_impulse_radii_flow_swollen,
    compute_agent_impulse_strengths,
    save_flow_field_viz_png,
)
from simulation.view_3d_mpl import save_cloth_3dview_png
from simulation.warp_springs import WarpSpringIntegrator


def main() -> None:
    ap = argparse.ArgumentParser(description="Warp CUDA mass-spring sheet (metal / cloth).")
    ap.add_argument("--device", type=str, default="cuda:0", help="Warp device, e.g. cuda:0 or cpu")
    ap.add_argument("--nu", type=int, default=128, help="Grid resolution in u (x)")
    ap.add_argument("--nv", type=int, default=None, help="Grid resolution in v (y along plane depth); default = nu")
    ap.add_argument("--plane-width", type=float, default=24.0)
    ap.add_argument("--plane-depth", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=500, help="Simulation frames at 60 Hz")
    ap.add_argument("--substeps", type=int, default=8, help="Spring substeps per frame")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument(
        "--gx",
        type=float,
        default=0.0,
        help="Gravity acceleration x (world units / s^2; default 0).",
    )
    ap.add_argument(
        "--gy",
        type=float,
        default=0.0,
        help="Gravity acceleration y (world units / s^2; default 0).",
    )
    ap.add_argument(
        "--gz",
        type=float,
        default=-9.81,
        help="Gravity acceleration z (world units / s^2; default -9.81 with Z up). Use 0 for zero gravity.",
    )
    ap.add_argument(
        "--velocity-drag",
        type=float,
        default=0.0,
        help="Exponential velocity damping per substep (1/s): v *= exp(-velocity_drag * dt). Reduces global sloshing / ringing (0=off).",
    )
    ap.add_argument("--stretch", type=float, default=2000.0)
    ap.add_argument("--bend", type=float, default=400.0)
    ap.add_argument("--shear", type=float, default=2000.0)
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument(
        "--anchor-strength",
        type=float,
        default=1e12,
        metavar="S",
        help="How firmly u=0 and u=nu-1 edges are held: edge particle mass is scaled by S vs interior "
        "(larger = stiffer). S >= 1e10 is treated as kinematic (fully fixed, default). Use ~10–1e6 "
        "for edges that can drift slightly under load.",
    )
    ap.add_argument(
        "--damping",
        type=float,
        default=10.0,
        help="Spring velocity damping coefficient (kd for each spring in the Warp integrator; default 10). Use 0 for undamped springs.",
    )
    ap.add_argument(
        "--plastic-yield",
        type=float,
        default=0.0,
        metavar="EPS",
        help="Spring plastic yield strain vs initial rest (0=disabled). E.g. 0.012 = 1.2%% elastic band; beyond that rest length creeps.",
    )
    ap.add_argument(
        "--plastic-creep",
        type=float,
        default=0.12,
        help="When yielded: fraction of excess |L-L0| (beyond yield band) applied to rest length per frame.",
    )
    ap.add_argument(
        "--plastic-max-strain",
        type=float,
        default=0.45,
        help="Clamp rest length to [(1-p)*L_init,(1+p)*L_init] per spring (p=this value).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/warp_sheet"))
    ap.add_argument("--disp-stride", type=int, default=10, help="Write displacement PNG every N frames")
    ap.add_argument("--export-size", type=int, default=512, help="PNG size (square); set 0 for native nu×nv")
    ap.add_argument("--impulse", action="store_true", help="Apply a small +Z velocity kick at sheet center (Z up)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--view3d",
        action="store_true",
        help="Save matplotlib 3D renders as 3dview_NNNNNN.png (see --view3d-stride).",
    )
    ap.add_argument(
        "--view3d-stride",
        type=int,
        default=1,
        metavar="N",
        help="When --view3d: write a PNG every N simulation frames (default 1 = every frame).",
    )
    ap.add_argument("--view3d-elev", type=float, default=45.0, help="Matplotlib view_init elev (deg)")
    ap.add_argument("--view3d-azim", type=float, default=45.0, help="Matplotlib view_init azim (deg)")
    ap.add_argument("--view3d-dpi", type=int, default=120, help="PNG resolution for 3dview captures")
    ap.add_argument(
        "--view3d-camera-dist",
        type=float,
        default=11.5,
        help="When --view3d: mplot3d camera distance (larger = more zoomed out; mpl default is often ~10).",
    )
    ap.add_argument(
        "--view3d-pad-inches",
        type=float,
        default=0.4,
        help="When --view3d: extra padding around the saved bbox (inches) so axes/ticks are not clipped.",
    )
    ap.add_argument(
        "--agents",
        action="store_true",
        help="Enable surface agents: fixed p5-style flow field for the run, periodic UV reseed, cloth impulses.",
    )
    ap.add_argument("--num-agents", type=int, default=12, metavar="N", help="With --agents: number of agents")
    ap.add_argument(
        "--agent-uv-margin",
        type=float,
        default=1.5,
        help="With --agents: hard UV clip from rim (grid cells); stay off anchored u-edges.",
    )
    ap.add_argument(
        "--agent-edge-keepout",
        type=float,
        default=4.0,
        help="With --agents: UV cells from sheet rim where inward repulsion ramps up (wider = less edge pooling).",
    )
    ap.add_argument(
        "--agent-edge-repel",
        type=float,
        default=0.22,
        help="With --agents: max inward UV nudge per frame at boundary (0 disables edge repulsion).",
    )
    ap.add_argument(
        "--agent-reseed-interval",
        type=int,
        default=20,
        metavar="N",
        help="With --agents: reseed agent UV every N frames (flow field stays fixed for the whole simulation).",
    )
    ap.add_argument(
        "--agent-impulse-period",
        type=int,
        default=5,
        metavar="N",
        help="With --agents: apply Z velocity impulse to cloth every N frames (after agent step, before integrate).",
    )
    ap.add_argument(
        "--agent-flow-damp",
        type=float,
        default=0.38,
        help="With --agents: flow smoothness in **sheet-normalized** XY (smaller = more dynamic twists; typical 0.15–0.65).",
    )
    ap.add_argument(
        "--agent-flow-scale",
        type=float,
        default=3.6,
        help="With --agents: spatial frequency scale into the noise grid (larger = finer structure on the sheet).",
    )
    ap.add_argument(
        "--agent-flow-grid",
        type=int,
        default=40,
        metavar="N",
        help="With --agents: resolution of the internal value-noise torus (N×N).",
    )
    ap.add_argument(
        "--agent-simplex-flow",
        action="store_true",
        help="With --agents: scale noise to 90° like p5 simplex2 mode (default: 180° Perlin-style).",
    )
    ap.add_argument(
        "--agent-step-cells",
        type=float,
        default=0.08,
        help="With --agents: agent advection step per frame in UV grid cell units.",
    )
    ap.add_argument(
        "--agent-impulse-strength",
        type=float,
        default=0.06,
        help="With --agents: peak velocity kick magnitude before mesh curvature scaling (±Z sign is fixed per agent at reseed).",
    )
    ap.add_argument(
        "--agent-impulse-radius",
        type=float,
        default=2.5,
        help="With --agents: hard UV cutoff (grid cells): vertices beyond this distance get no kick.",
    )
    ap.add_argument(
        "--agent-impulse-sigma-frac",
        type=float,
        default=0.22,
        help="With --agents: Gaussian std = radius * this, inside the disk only (smaller = sharper local peak).",
    )
    ap.add_argument(
        "--agent-impulse-radius-jitter",
        type=float,
        default=0.1,
        metavar="F",
        help="With --agents: per-agent UV impulse radius is uniform in [R*(1-F), R*(1+F)] on each reseed (0=no jitter).",
    )
    ap.add_argument(
        "--agent-impulse-curvature-grade",
        type=float,
        default=0.2,
        metavar="F",
        help="With --agents: scale each alive agent's peak impulse by local mesh Laplacian vs peers: "
        "strength in [S*(1-F), S*(1+F)] where S is --agent-impulse-strength (F=0 disables).",
    )
    ap.add_argument(
        "--agent-flow-radius-swell",
        type=float,
        default=0.1,
        metavar="F",
        help="With --agents: in high flow-angle spatial curvature, swell UV impulse radius by up to "
        "this fraction (e.g. 0.1 → up to +10%% over the per-agent base radius; 0 disables).",
    )
    ap.add_argument(
        "--agent-impulse-normal-tilt",
        type=float,
        default=0.2,
        metavar="F",
        help="With --agents: max fraction blending ±Z toward the local deformed mesh normal at the agent "
        "(0 = pure ±Z). Ramps from 0 on a flat sheet to F as RMS Z reaches --agent-impulse-tilt-rms-ref.",
    )
    ap.add_argument(
        "--agent-impulse-tilt-rms-ref",
        type=float,
        default=None,
        metavar="L",
        help="With --agents: RMS Z of vertex positions at which normal-tilt reaches full "
        "--agent-impulse-normal-tilt (default: 0.025 * max(plane width, depth)). Must be > 0 if set.",
    )
    ap.add_argument(
        "--viz-flow-field",
        action="store_true",
        help="Save a quiver PNG of the agent flow field (uses --plane-* and --agent-flow-*; honors --seed).",
    )
    ap.add_argument(
        "--viz-flow-field-only",
        action="store_true",
        help="With --viz-flow-field: write the PNG and exit (no cloth / Warp simulation).",
    )
    ap.add_argument(
        "--viz-flow-field-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="With --viz-flow-field: output PNG path (default: <out-dir>/flow_field_viz.png).",
    )
    ap.add_argument(
        "--viz-flow-field-grid",
        type=int,
        default=72,
        metavar="N",
        help="With --viz-flow-field: N×N samples in normalized sheet coordinates.",
    )
    ap.add_argument(
        "--viz-flow-field-dpi",
        type=int,
        default=140,
        help="With --viz-flow-field: PNG DPI.",
    )
    args = ap.parse_args()

    nv = args.nv if args.nv is not None else args.nu
    if args.nu < 3 or nv < 3:
        raise SystemExit("--nu and --nv must be >= 3")
    if args.view3d_stride < 1:
        raise SystemExit("--view3d-stride must be >= 1")
    if args.view3d_camera_dist <= 0.0:
        raise SystemExit("--view3d-camera-dist must be > 0")
    if args.view3d_pad_inches < 0.0:
        raise SystemExit("--view3d-pad-inches must be >= 0")
    if args.viz_flow_field_only and not args.viz_flow_field:
        raise SystemExit("--viz-flow-field-only requires --viz-flow-field")
    if args.viz_flow_field and args.viz_flow_field_grid < 8:
        raise SystemExit("--viz-flow-field-grid must be >= 8")
    if args.agents or args.viz_flow_field:
        if args.agent_flow_damp <= 0.0:
            raise SystemExit("--agent-flow-damp must be > 0")
        if args.agent_flow_scale <= 0.0:
            raise SystemExit("--agent-flow-scale must be > 0")
        if args.agent_flow_grid < 4:
            raise SystemExit("--agent-flow-grid must be >= 4")
    if args.agents:
        if args.num_agents < 1:
            raise SystemExit("--num-agents must be >= 1")
        if args.agent_reseed_interval < 1:
            raise SystemExit("--agent-reseed-interval must be >= 1")
        if args.agent_impulse_period < 1:
            raise SystemExit("--agent-impulse-period must be >= 1")
        if args.agent_impulse_sigma_frac <= 0.0:
            raise SystemExit("--agent-impulse-sigma-frac must be > 0")
        pad = max(float(args.agent_uv_margin), float(args.agent_edge_keepout))
        if 2.0 * pad >= float(min(args.nu, nv)) - 1.0:
            raise SystemExit(
                "agents: require min(nu,nv) > 2*max(--agent-uv-margin, --agent-edge-keepout)+1; "
                "lower edge keepout/margin or use a finer grid."
            )
        if args.agent_edge_repel < 0.0:
            raise SystemExit("--agent-edge-repel must be >= 0")
        if args.agent_impulse_radius_jitter < 0.0 or args.agent_impulse_radius_jitter > 0.95:
            raise SystemExit("--agent-impulse-radius-jitter must be in [0, 0.95]")
        if args.agent_impulse_curvature_grade < 0.0 or args.agent_impulse_curvature_grade > 1.0:
            raise SystemExit("--agent-impulse-curvature-grade must be in [0, 1]")
        if args.agent_flow_radius_swell < 0.0 or args.agent_flow_radius_swell > 1.0:
            raise SystemExit("--agent-flow-radius-swell must be in [0, 1]")
        if args.agent_impulse_normal_tilt < 0.0 or args.agent_impulse_normal_tilt > 1.0:
            raise SystemExit("--agent-impulse-normal-tilt must be in [0, 1]")
        if args.agent_impulse_tilt_rms_ref is not None and args.agent_impulse_tilt_rms_ref <= 0.0:
            raise SystemExit("--agent-impulse-tilt-rms-ref must be > 0 when set")
    if args.viz_flow_field and args.viz_flow_field_dpi < 1:
        raise SystemExit("--viz-flow-field-dpi must be >= 1")
    if args.plastic_yield < 0.0:
        raise SystemExit("--plastic-yield must be >= 0")
    if args.plastic_creep < 0.0 or args.plastic_creep > 1.0:
        raise SystemExit("--plastic-creep must be in [0, 1]")
    if args.plastic_max_strain <= 0.0:
        raise SystemExit("--plastic-max-strain must be > 0")
    if args.mass <= 0.0:
        raise SystemExit("--mass must be > 0")
    if args.anchor_strength < 1.0:
        raise SystemExit("--anchor-strength must be >= 1")
    if args.damping < 0.0:
        raise SystemExit("--damping must be >= 0")
    if args.velocity_drag < 0.0:
        raise SystemExit("--velocity-drag must be >= 0")

    wp.init()
    rng = np.random.default_rng(args.seed)

    flow_field: P5StyleFlowField | None = None
    if args.agents or args.viz_flow_field:
        fp = FlowFieldParams(
            plane_width=float(args.plane_width),
            plane_depth=float(args.plane_depth),
            damp_x=float(args.agent_flow_damp),
            damp_y=float(args.agent_flow_damp),
            spatial_scale=float(args.agent_flow_scale),
            grid_n=int(args.agent_flow_grid),
            simplex_style=bool(args.agent_simplex_flow),
        )
        flow_field = P5StyleFlowField(fp)
        flow_field.regenerate(rng)
        if args.viz_flow_field:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            outv = args.viz_flow_field_out
            if outv is None:
                outv = args.out_dir / "flow_field_viz.png"
            save_flow_field_viz_png(
                outv,
                flow_field,
                grid=int(args.viz_flow_field_grid),
                dpi=int(args.viz_flow_field_dpi),
            )
            print(f"Wrote flow field viz: {outv}")
        if args.viz_flow_field_only:
            print("Exiting (--viz-flow-field-only).")
            raise SystemExit(0)

    cloth = ClothGrid(
        args.nu,
        nv,
        plane_width=args.plane_width,
        plane_depth=args.plane_depth,
        stretch_stiffness=args.stretch,
        bend_stiffness=args.bend,
        shear_stiffness=args.shear,
        mass=args.mass,
        spring_damping=args.damping,
        anchor_u_edges=True,
        anchor_strength=float(args.anchor_strength),
    )
    rest = cloth.rest_positions()
    disp_half_extent = fixed_domain_half_extent_from_rest(rest)
    tri_flat = np.asarray(cloth.triangles, dtype=np.int32)

    try:
        integrator = WarpSpringIntegrator(
            cloth,
            device=args.device,
            gravity=(args.gx, args.gy, args.gz),
            velocity_drag=args.velocity_drag,
            plastic_yield_strain=args.plastic_yield,
            plastic_creep=args.plastic_creep,
            plastic_max_strain=args.plastic_max_strain,
        )
    except Exception as e:
        print(f"Failed to create integrator on {args.device}: {e}", file=sys.stderr)
        if args.device != "cpu":
            print("Retry with --device cpu", file=sys.stderr)
        raise SystemExit(1) from e

    if args.impulse:
        integrator.add_velocity_impulse(
            args.nu // 2,
            nv // 2,
            radius=float(min(args.nu, nv)) * 0.12,
            strength=0.35 + 0.15 * rng.random(),
        )

    agent_swarm: MeshFlowAgentSwarm | None = None
    if args.agents:
        assert flow_field is not None
        agent_swarm = MeshFlowAgentSwarm(
            args.num_agents,
            args.nu,
            nv,
            margin_cells=float(args.agent_uv_margin),
            edge_keepout_cells=float(args.agent_edge_keepout),
            edge_repel_strength=float(args.agent_edge_repel),
            impulse_radius_base=float(args.agent_impulse_radius),
            impulse_radius_jitter_frac=float(args.agent_impulse_radius_jitter),
        )

    dt = 1.0 / args.fps
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_size = None if args.export_size == 0 else int(args.export_size)

    disp_idx = 0
    view3d_idx = 0
    warned_bad = False
    for frame in range(1, args.steps + 1):
        pos_pre = integrator.numpy_positions()
        if args.agents and flow_field is not None and agent_swarm is not None:
            if (frame - 1) % args.agent_reseed_interval == 0:
                agent_swarm.reseed(rng)
            agent_swarm.step(pos_pre, flow_field, args.agent_step_cells)
            if frame % args.agent_impulse_period == 0:
                mags = compute_agent_impulse_strengths(
                    pos_pre,
                    args.nu,
                    nv,
                    agent_swarm.uv,
                    agent_swarm.agent_alive,
                    float(args.agent_impulse_strength),
                    curvature_grade_frac=float(args.agent_impulse_curvature_grade),
                )
                rms_z = float(np.sqrt(np.mean(np.square(pos_pre[:, 2]))))
                tilt_ref = (
                    float(args.agent_impulse_tilt_rms_ref)
                    if args.agent_impulse_tilt_rms_ref is not None
                    else (0.025 * max(float(args.plane_width), float(args.plane_depth)))
                )
                dirs = compute_agent_impulse_directions(
                    pos_pre,
                    args.nu,
                    nv,
                    agent_swarm.uv,
                    agent_swarm.impulse_z_sign,
                    mags,
                    tilt_max=float(args.agent_impulse_normal_tilt),
                    deform_rms_z=rms_z,
                    deform_rms_ref=tilt_ref,
                )
                radii_uv = compute_agent_impulse_radii_flow_swollen(
                    flow_field,
                    pos_pre,
                    args.nu,
                    nv,
                    agent_swarm.uv,
                    agent_swarm.agent_alive,
                    agent_swarm.impulse_radius_uv,
                    swell_frac=float(args.agent_flow_radius_swell),
                )
                integrator.add_velocity_impulses_uv_gaussian(
                    agent_swarm.uv,
                    mags,
                    radii_uv,
                    args.nu,
                    nv,
                    sigma_frac=args.agent_impulse_sigma_frac,
                    impulse_dir_world=dirs,
                )
        integrator.simulate(dt, args.substeps)
        pos = integrator.numpy_positions()
        if not warned_bad and not np.all(np.isfinite(pos)):
            print(
                "warning: non-finite vertex positions (NaN/Inf); integrator may be unstable. "
                "Try smaller dt via --fps 120 (or higher), more --substeps, lower --stretch, or a weaker --impulse. "
                "With --agents, try lower --agent-impulse-strength, smaller --agent-impulse-radius, "
                "smaller --agent-impulse-sigma-frac for a sharper bump, or --velocity-drag > 0. "
                "Try higher --damping (spring kd) if oscillations persist. "
                "Displacement PNGs still write with neutral bytes for bad samples.",
                file=sys.stderr,
            )
            warned_bad = True

        if args.view3d and frame % args.view3d_stride == 0:
            vpath = args.out_dir / f"3dview_{view3d_idx:06d}.png"
            save_cloth_3dview_png(
                pos,
                tri_flat,
                vpath,
                elev_deg=args.view3d_elev,
                azim_deg=args.view3d_azim,
                dpi=int(args.view3d_dpi),
                camera_dist=float(args.view3d_camera_dist),
                save_pad_inches=float(args.view3d_pad_inches),
            )
            print(f"frame {frame}/{args.steps} wrote {vpath}")
            view3d_idx += 1

        if frame % args.disp_stride != 0:
            continue
        path = args.out_dir / f"vertex-displacement-rgb-warp_{disp_idx:06d}.png"
        save_displacement_png(
            str(path),
            pos,
            rest,
            args.nu,
            nv,
            out_size=out_size,
            d=disp_half_extent,
        )
        print(f"frame {frame}/{args.steps} wrote {path}")
        disp_idx += 1

    final = integrator.numpy_positions()
    out_npz: dict = {"positions": final, "rest": rest, "nu": np.int32(args.nu), "nv": np.int32(nv)}
    if args.plastic_yield > 0.0:
        out_npz["spring_rest_lengths"] = integrator.numpy_spring_rest_lengths()
        out_npz["plastic_yield_strain"] = np.float64(args.plastic_yield)
        out_npz["plastic_creep"] = np.float64(args.plastic_creep)
        out_npz["plastic_max_strain"] = np.float64(args.plastic_max_strain)
    np.savez_compressed(args.out_dir / "sheet_final.npz", **out_npz)
    print("Done. Summary:", args.out_dir)


if __name__ == "__main__":
    main()
