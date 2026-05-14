#!/usr/bin/env python3
"""
Batch-launch ``python -m simulation.run_sheet_warp`` with a shared baseline argv, unique ``--seed``
and ``--out-dir`` per run, and optional concurrent subprocess workers.

From the **repository root** (parent of the ``simulation`` package):

  python -m simulation.batch_run_sheet_warp --runs 8 --jobs 4

Override any baseline flag by appending argparse tokens **after** the batch flags (unknown tokens
are forwarded and override earlier duplicates for most ``run_sheet_warp`` options):

  python -m simulation.batch_run_sheet_warp --runs 3 --jobs 1 -- --steps 100 --nu 64

Use ``--dry-run`` to print commands without executing. On a **single GPU**, keep ``--jobs 1`` unless
you know concurrent CUDA runs are safe for your setup.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repository root: .../ACADIA-2026 (parent of ``simulation/``)
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Baseline argv for ``run_sheet_warp`` (no ``--seed``, no ``--out-dir`` — those are set per run).
_DEFAULT_WARP_ARGV: list[str] = [
    "-m",
    "simulation.run_sheet_warp",
    "--device",
    "cuda:0",
    "--nu",
    "128",
    "--nv",
    "128",
    "--plane-width",
    "24",
    "--plane-depth",
    "24",
    "--steps",
    "500",
    "--substeps",
    "16",
    "--fps",
    "60",
    "--stretch",
    "4000",
    "--bend",
    "2000",
    "--shear",
    "2000",
    "--mass",
    "1",
    "--disp-stride",
    "200",
    "--export-size",
    "512",
    "--view3d",
    "--view3d-stride",
    "36",
    "--view3d-elev",
    "45",
    "--view3d-azim",
    "45",
    "--view3d-dpi",
    "300",
    "--impulse",
    "--agents",
    "--num-agents",
    "12",
    "--agent-reseed-interval",
    "36",
    "--agent-impulse-period",
    "1",
    "--agent-flow-damp",
    "200.0",
    "--agent-flow-scale",
    "320.0",
    "--agent-flow-grid",
    "40",
    "--agent-simplex-flow",
    "--agent-step-cells",
    "4",
    "--agent-impulse-strength",
    "40",
    "--agent-impulse-radius",
    "40",
    "--agent-impulse-sigma-frac",
    "0.14",
    "--plastic-yield",
    "0.002",
    "--plastic-creep",
    "0.85",
    "--plastic-max-strain",
    "0.85",
    "--gz",
    "-0.1",
    "--damping",
    "100",
    "--velocity-drag",
    "100",
    "--agent-edge-keepout",
    "2.0",
    "--agent-edge-repel",
    "50",
    "--agent-impulse-radius-jitter",
    "0.4",
    "--agent-impulse-curvature-grade",
    "0.8",
    "--agent-flow-radius-swell",
    "0.8",
    "--anchor-strength",
    "10",
    "--agent-impulse-normal-tilt",
    "0.6",
]


def _strip_seed_and_out_dir(tokens: list[str]) -> list[str]:
    """Remove ``--seed`` / ``--out-dir`` and their values so per-run values can be appended last."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--seed", "--out-dir"):
            i += 2
            continue
        if t.startswith("--seed=") or t.startswith("--out-dir="):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _build_cmd(
    *,
    seed: int,
    out_dir: Path,
    warp_extra: list[str],
    device: str | None,
) -> list[str]:
    parts = list(_DEFAULT_WARP_ARGV)
    if device is not None:
        try:
            di = parts.index("--device")
            parts[di + 1] = device
        except ValueError:
            parts[1:1] = ["--device", device]
    extra = _strip_seed_and_out_dir(warp_extra)
    cmd = [sys.executable] + parts + extra
    cmd += ["--seed", str(int(seed)), "--out-dir", str(out_dir)]
    return cmd


def _run_one(
    *,
    run_index: int,
    seed: int,
    out_dir_base: Path,
    name_prefix: str,
    warp_extra: list[str],
    device: str | None,
    dry_run: bool,
) -> tuple[int, int, int, str]:
    """Returns (run_index, seed, returncode, out_dir_str)."""
    out_dir = out_dir_base / f"{name_prefix}_{run_index:04d}_seed_{seed}"
    cmd = _build_cmd(seed=seed, out_dir=out_dir, warp_extra=warp_extra, device=device)
    printable = " ".join(f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd)
    if dry_run:
        print(f"[dry-run] run {run_index} seed={seed} -> {out_dir}")
        print(f"  {printable}")
        return run_index, seed, 0, str(out_dir)
    out_dir_base.mkdir(parents=True, exist_ok=True)
    print(f"[start] run {run_index} seed={seed} -> {out_dir}", flush=True)
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=os.environ.copy())
    status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
    print(f"[done]  run {run_index} seed={seed} {status}", flush=True)
    return run_index, seed, int(r.returncode), str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch run simulation.run_sheet_warp with varying --seed and --out-dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-n", "--runs", type=int, required=True, metavar="N", help="Total number of runs")
    ap.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Seed for run index 0; run i uses seed_start + i.",
    )
    ap.add_argument(
        "--out-dir-base",
        type=Path,
        default=Path("data/warp_sheet_batch"),
        help="Directory under repo root; each run writes to a subfolder run_XXXX_seed_S.",
    )
    ap.add_argument(
        "--name-prefix",
        type=str,
        default="run",
        help="Subfolder name prefix: {prefix}_{i:04d}_seed_{seed}.",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="J",
        help="Max concurrent subprocesses (use 1 on a single GPU unless you accept contention / OOM).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="If set, overrides the --device value in the baseline argv for every run.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    args, warp_extra = ap.parse_known_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    out_base = args.out_dir_base
    if not out_base.is_absolute():
        out_base = _REPO_ROOT / out_base

    seeds = [int(args.seed_start + i) for i in range(args.runs)]
    print(
        f"Batch: {args.runs} runs, seeds {seeds[0]}..{seeds[-1]}, out_dir_base={out_base}, jobs={args.jobs}",
        flush=True,
    )

    results: list[tuple[int, int, int, str]] = []
    if args.jobs == 1:
        for i in range(args.runs):
            results.append(
                _run_one(
                    run_index=i,
                    seed=seeds[i],
                    out_dir_base=out_base,
                    name_prefix=args.name_prefix,
                    warp_extra=warp_extra,
                    device=args.device,
                    dry_run=args.dry_run,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [
                ex.submit(
                    _run_one,
                    run_index=i,
                    seed=seeds[i],
                    out_dir_base=out_base,
                    name_prefix=args.name_prefix,
                    warp_extra=warp_extra,
                    device=args.device,
                    dry_run=args.dry_run,
                )
                for i in range(args.runs)
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda t: t[0])
    failed = [t for t in results if t[2] != 0]
    print("--- summary ---", flush=True)
    for run_i, seed, code, od in results:
        print(f"  run {run_i:04d} seed={seed} code={code} out={od}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed with non-zero exit code.")


if __name__ == "__main__":
    main()
