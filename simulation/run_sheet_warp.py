#!/usr/bin/env python3
"""
CUDA sheet / cloth simulation using NVIDIA Warp (mass–spring grid).

Upstream reference (Apache-2.0):
  https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth.py
  https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth_warp.py

Run (from repo root):
  python -m simulation.run_sheet_warp --steps 300 --device cuda:0 --out-dir data/warp_run1

Requires: pip install warp-lang numpy pillow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(_REPO))

import warp as wp

from simulation.cloth_grid import ClothGrid
from simulation.displacement_png import save_displacement_png
from simulation.warp_springs import WarpSpringIntegrator


def main() -> None:
    ap = argparse.ArgumentParser(description="Warp CUDA mass-spring sheet (metal / cloth).")
    ap.add_argument("--device", type=str, default="cuda:0", help="Warp device, e.g. cuda:0 or cpu")
    ap.add_argument("--nu", type=int, default=128, help="Grid resolution in u (x)")
    ap.add_argument("--nv", type=int, default=None, help="Grid resolution in v (z); default = nu")
    ap.add_argument("--plane-width", type=float, default=24.0)
    ap.add_argument("--plane-depth", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=500, help="Simulation frames at 60 Hz")
    ap.add_argument("--substeps", type=int, default=8, help="Spring substeps per frame")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--stretch", type=float, default=2000.0)
    ap.add_argument("--bend", type=float, default=400.0)
    ap.add_argument("--shear", type=float, default=2000.0)
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, default=Path("data/warp_sheet"))
    ap.add_argument("--disp-stride", type=int, default=10, help="Write displacement PNG every N frames")
    ap.add_argument("--export-size", type=int, default=512, help="PNG size (square); set 0 for native nu×nv")
    ap.add_argument("--impulse", action="store_true", help="Apply a small +Y velocity kick at sheet center")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    nv = args.nv if args.nv is not None else args.nu
    if args.nu < 3 or nv < 3:
        raise SystemExit("--nu and --nv must be >= 3")

    wp.init()
    rng = np.random.default_rng(args.seed)

    cloth = ClothGrid(
        args.nu,
        nv,
        plane_width=args.plane_width,
        plane_depth=args.plane_depth,
        stretch_stiffness=args.stretch,
        bend_stiffness=args.bend,
        shear_stiffness=args.shear,
        mass=args.mass,
        anchor_u_edges=True,
    )
    rest = cloth.rest_positions()

    try:
        integrator = WarpSpringIntegrator(cloth, device=args.device)
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

    dt = 1.0 / args.fps
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_size = None if args.export_size == 0 else int(args.export_size)

    disp_idx = 0
    for frame in range(1, args.steps + 1):
        integrator.simulate(dt, args.substeps)
        if frame % args.disp_stride != 0:
            continue
        pos = integrator.numpy_positions()
        path = args.out_dir / f"vertex-displacement-rgb-warp_{disp_idx:06d}.png"
        save_displacement_png(
            str(path),
            pos,
            rest,
            args.nu,
            nv,
            out_size=out_size,
            d=None,
        )
        print(f"frame {frame}/{args.steps} wrote {path}")
        disp_idx += 1

    final = integrator.numpy_positions()
    np.savez_compressed(args.out_dir / "sheet_final.npz", positions=final, rest=rest, nu=args.nu, nv=nv)
    print("Done. Summary:", args.out_dir)


if __name__ == "__main__":
    main()
