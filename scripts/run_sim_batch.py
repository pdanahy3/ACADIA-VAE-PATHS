#!/usr/bin/env python3
"""
Headless batch runs of the sheet-metal WebGL simulation.

Starts serve_capture.py if nothing is listening on the port, then drives batch.html
via Playwright: each iteration uses a random seed and random boid count (6–32),
runs --timesteps simulation steps, then POSTs captures under DATA_DIR/<folder>/.
Use -j/--workers (1–16) to run iterations across independent Chromium processes in parallel.
Cadence defaults: view every 50 steps, displacement every 2; override with
--view-stride and --disp-stride (passed as URL view_stride / disp_stride).

Folder name encodes iteration index and seed.

Install once (use python -m so it works without a Playwright CLI on PATH):
  pip install -r requirements-batch.txt
  python -m playwright install chromium

Or from this repo:
  python scripts/run_sim_batch.py --install-browsers

Example (from repo root):
  python scripts/run_sim_batch.py -n 5 -t 400
  python scripts/run_sim_batch.py -n 5 -t 400 --webgpu
  python scripts/run_sim_batch.py -n 5 -t 400 --view-stride 100 --disp-stride 5
  python scripts/run_sim_batch.py -n 32 -t 200 -j 8
"""
from __future__ import annotations

import argparse
import os
import random
import socket
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _ensure_server(host: str, port: int) -> subprocess.Popen | None:
    if _port_open(host, port):
        print(f"Using existing server on http://{host}:{port}/")
        return None
    print(f"Starting capture server on http://{host}:{port}/ ...")
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / "serve_capture.py")],
        cwd=str(SCRIPT_DIR),
    )
    for _ in range(80):
        if _port_open(host, port):
            time.sleep(0.15)
            return proc
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("serve_capture.py did not become reachable in time")


def _install_playwright_browsers() -> None:
    print("Running: python -m playwright install chromium")
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        cwd=str(SCRIPT_DIR),
        check=True,
    )
    print("Chromium for Playwright is installed.")


def _make_batch_url(
    base: str,
    timesteps: int,
    *,
    folder: str,
    seed: int,
    boids: int,
    view_stride: int,
    disp_stride: int,
    webgpu: bool,
) -> str:
    qs_dict = {
        "batch": "1",
        "steps": str(timesteps),
        "folder": folder,
        "seed": str(seed),
        "boids": str(boids),
        "view_stride": str(view_stride),
        "disp_stride": str(disp_stride),
    }
    if webgpu:
        qs_dict["webgpu"] = "1"
    return f"{base}?{urllib.parse.urlencode(qs_dict)}"


def _split_iteration_buckets(specs: list[dict], n_buckets: int) -> list[list[dict]]:
    """Round-robin split so each bucket has similar load."""
    if not specs:
        return []
    n_buckets = max(1, min(n_buckets, len(specs)))
    buckets: list[list[dict]] = [[] for _ in range(n_buckets)]
    for i, s in enumerate(specs):
        buckets[i % n_buckets].append(s)
    return buckets


def _playwright_worker_chunk(payload: dict) -> list[dict]:
    """
    Run multiple batch iterations sequentially in one Chromium instance (separate OS process).
    """
    from playwright.sync_api import sync_playwright

    host = payload["host"]
    port = payload["port"]
    timesteps = payload["timesteps"]
    webgpu = payload["webgpu"]
    view_stride = payload["view_stride"]
    disp_stride = payload["disp_stride"]
    iterations = payload["iterations"]
    total_runs = payload["total_runs"]
    base = f"http://{host}:{port}/batch.html"
    pid = os.getpid()
    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            _attach_batch_console(page)
            for it in iterations:
                url = _make_batch_url(
                    base,
                    timesteps,
                    folder=it["folder"],
                    seed=it["seed"],
                    boids=it["boids"],
                    view_stride=view_stride,
                    disp_stride=disp_stride,
                    webgpu=webgpu,
                )
                r = it["index"]
                print(
                    f"[worker pid {pid}] iteration {r + 1}/{total_runs} folder={it['folder']}"
                )
                page.goto(url, wait_until="load", timeout=1_800_000)
                page.wait_for_function(
                    "() => window.__BATCH_DONE__ !== undefined",
                    timeout=1_800_000,
                )
                done = page.evaluate("() => window.__BATCH_DONE__")
                print(f"  [worker pid {pid}] {it['folder']}: {done}")
                if not isinstance(done, dict) or not done.get("ok"):
                    raise RuntimeError(f"Batch failed for {it['folder']}: {done!r}")
                results.append({"folder": it["folder"], "done": done})
        finally:
            browser.close()
    return results


def _attach_batch_console(page) -> None:
    """Echo in-page `[batch] ...` logs to the terminal (headless progress)."""

    def _on_console(msg) -> None:
        try:
            text = msg.text
        except Exception:
            text = str(msg)
        if "[batch]" in text:
            print(f"  {text}", flush=True)

    page.on("console", _on_console)

    def _on_page_error(exc) -> None:
        print(f"  [pageerror] {exc}", flush=True)

    page.on("pageerror", _on_page_error)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run N batch iterations with T timesteps each (Playwright + batch.html)."
    )
    ap.add_argument(
        "-n",
        "--runs",
        type=int,
        default=None,
        help="Number of iterations (each with a new random seed and boid count 6–32).",
    )
    ap.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=None,
        help="Simulation timesteps per iteration (captures on steps divisible by --view-stride / --disp-stride).",
    )
    ap.add_argument(
        "--install-browsers",
        action="store_true",
        help="Download Playwright Chromium (same as: python -m playwright install chromium), then exit unless -n/-t are also set.",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port for serve_capture.py (default 8765).",
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local server (default 127.0.0.1).",
    )
    ap.add_argument(
        "--webgpu",
        action="store_true",
        help="Append webgpu=1 to the batch URL (GPU cloth physics in Chromium when supported).",
    )
    ap.add_argument(
        "--view-stride",
        type=int,
        default=50,
        metavar="N",
        help="Save view (render) capture every N simulation steps (URL: view_stride). Default 50.",
    )
    ap.add_argument(
        "--disp-stride",
        type=int,
        default=2,
        metavar="N",
        help="Save displacement PNG every N simulation steps (URL: disp_stride). Default 2.",
    )
    ap.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="W",
        help="Parallel batch workers (separate Chromium processes), 1–16. Iterations are split across workers.",
    )
    args = ap.parse_args()

    if args.install_browsers:
        _install_playwright_browsers()
        if args.runs is None or args.timesteps is None:
            return

    if args.runs is None or args.timesteps is None:
        ap.error("the following arguments are required: -n/--runs, -t/--timesteps")

    if args.runs < 1 or args.timesteps < 1:
        print("runs and timesteps must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.view_stride < 1 or args.disp_stride < 1:
        print("view-stride and disp-stride must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.workers < 1 or args.workers > 16:
        print("workers must be between 1 and 16", file=sys.stderr)
        sys.exit(2)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Missing Playwright. Run:\n"
            f"  pip install -r {SCRIPT_DIR / 'requirements-batch.txt'}\n"
            "  python -m playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(1)

    proc: subprocess.Popen | None = None
    try:
        proc = _ensure_server(args.host, args.port)
        base = f"http://{args.host}:{args.port}/batch.html"

        specs: list[dict] = []
        for r in range(args.runs):
            seed = random.randint(1, 2**31 - 1)
            boids = random.randint(6, 32)
            folder = f"iter_{r:04d}_seed_{seed}_b{boids}"
            specs.append({"index": r, "seed": seed, "boids": boids, "folder": folder})

        n_parallel = max(1, min(args.workers, 16, args.runs))

        if n_parallel == 1:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True)
                except Exception as e:
                    err = str(e).lower()
                    if "executable" in err or "doesn't exist" in err:
                        print(
                            "Playwright browser binaries are missing. Install with:\n"
                            f"  {sys.executable} -m playwright install chromium\n"
                            "Or run:\n"
                            f"  {sys.executable} {Path(__file__).resolve()} --install-browsers",
                            file=sys.stderr,
                        )
                    raise
                try:
                    page = browser.new_page(viewport={"width": 1920, "height": 1080})
                    _attach_batch_console(page)
                    for it in specs:
                        url = _make_batch_url(
                            base,
                            args.timesteps,
                            folder=it["folder"],
                            seed=it["seed"],
                            boids=it["boids"],
                            view_stride=args.view_stride,
                            disp_stride=args.disp_stride,
                            webgpu=args.webgpu,
                        )
                        r = it["index"]
                        print(f"Iteration {r + 1}/{args.runs} folder={it['folder']}")
                        page.goto(url, wait_until="load", timeout=1_800_000)
                        print(
                            f"  …running {args.timesteps} steps; watch lines starting with [batch] for progress; "
                            "first POST to /api/save-capture happens after step disp_stride (default 2)."
                        )
                        page.wait_for_function(
                            "() => window.__BATCH_DONE__ !== undefined",
                            timeout=1_800_000,
                        )
                        done = page.evaluate("() => window.__BATCH_DONE__")
                        print(" ", done)
                        if not isinstance(done, dict) or not done.get("ok"):
                            raise RuntimeError(f"Batch failed: {done!r}")
                finally:
                    browser.close()
        else:
            buckets = _split_iteration_buckets(specs, n_parallel)
            payloads = [
                {
                    "host": args.host,
                    "port": args.port,
                    "timesteps": args.timesteps,
                    "webgpu": args.webgpu,
                    "view_stride": args.view_stride,
                    "disp_stride": args.disp_stride,
                    "iterations": bucket,
                    "total_runs": args.runs,
                }
                for bucket in buckets
            ]
            n_disp = args.timesteps // args.disp_stride
            n_view = args.timesteps // args.view_stride
            print(
                f"Running {args.runs} iterations across {len(payloads)} parallel workers "
                f"(-j/--workers {args.workers}, effective {len(payloads)}).\n"
                f"  Each iteration can trigger up to ~{n_disp} displacement + ~{n_view} view capture POSTs "
                f"(disp_stride={args.disp_stride}, view_stride={args.view_stride}) — heavy; first POST is after step {args.disp_stride}.\n"
                "  Progress: lines starting with `[batch]` echo from headless Chromium. This log shows "
                "`POST /api/save-capture` when PNGs are written."
            )
            with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
                futures = [pool.submit(_playwright_worker_chunk, pl) for pl in payloads]
                for fut in as_completed(futures):
                    fut.result()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("All iterations finished.")


if __name__ == "__main__":
    main()
