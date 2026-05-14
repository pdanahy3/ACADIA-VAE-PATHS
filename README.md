# ACADIA-VAE-PATHS

Primary **physics** path: **Python + NVIDIA Warp on CUDA** (`simulation/`) for a mass–spring sheet with displacement PNG export aligned to the browser pipeline. Optional **Three.js** viewer and headless capture remain under `scripts/` for visualization and legacy workflows. **VAE** training and **t-SNE** use the displacement images.

## Environment

Use conda (recommended) or a virtual environment.

```bash
conda activate ACADIA2026
pip install -r requirements.txt
```

**CUDA:** install an NVIDIA driver compatible with your GPU. Warp bundles a CUDA toolkit for JIT; `python -c "import warp as wp; wp.init()"` prints the toolkit and driver it detected (e.g. CUDA 12.x runtime with a 13.x driver is normal).

For headless **browser** batch capture only (Playwright, no Warp), you can use `scripts/requirements-batch.txt` instead of the full ML stack if you prefer a lighter install.

## Simulation (Python + Warp, CUDA)

This replaces the heavy per-frame browser physics for **data generation** with a single Python entry point. It is adapted from NVIDIA’s cloth benchmark:

- [benchmark_cloth.py](https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth.py) (grid + springs)
- [benchmark_cloth_warp.py](https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark_cloth_warp.py) (Warp kernels + integrator)

Apache-2.0; SPDX headers are kept in `simulation/cloth_grid.py` and `simulation/warp_springs.py`.

From the **repository root** (use the **same** Python as your conda env; install **`warp-lang`**, not `warp`):

```bash
python -m pip install warp-lang
python -m simulation.run_sheet_warp --device cuda:0 --steps 400 --nu 128 --disp-stride 10 --impulse --out-dir data/warp_run1
python -m simulation.run_sheet_warp --device cuda:0 --steps 200 --nu 96 --view3d --view3d-stride 1 --disp-stride 20 --out-dir data/warp_views
```

The PyPI package **`warp`** is unrelated and will not provide `import warp` for NVIDIA Warp. If you installed it by mistake: `python -m pip uninstall warp` then install **`warp-lang`**.

- **`--nu` / `--nv`**: grid size (default 128×128). Full **512×512** is much heavier (more particles and springs); start smaller, or use `--export-size 512` to upsample PNGs for the VAE.
- **`--disp-stride`**: how often to write `vertex-displacement-rgb-warp_*.png`.
- **`--impulse`**: optional velocity kick so the sheet is not static.

## Browser viewer (optional Three.js)

Serve the `scripts` folder over HTTP (module imports require a server, not `file://`).

```bash
cd scripts
python serve_capture.py
```

Open the printed URL (for example `http://127.0.0.1:8765/sheet-metal-simulation.html`).

### Query flags

| URL flag | Effect |
|----------|--------|
| `?webgpu=1` | Run Verlet integration and stretch/bend constraints on the **GPU** (WebGPU). Falls back to CPU if initialization fails. Plasticity, boids, and impulses still run on the CPU; state is uploaded each step. |
| `?batch=1` | Headless batch mode: steps the sim and POSTs captures to the capture API (see `run_sim_batch.py`). Combine with `&webgpu=1` if the browser supports WebGPU for batch runs. |
| `view_stride=N` | Save **view** (render) captures every **N** simulation steps when capturing or in batch mode (default **50** if omitted or invalid). |
| `disp_stride=N` | Save **displacement** PNGs every **N** simulation steps (default **2**). |

Example:

```text
http://127.0.0.1:8765/sheet-metal-simulation.html?webgpu=1
http://127.0.0.1:8765/sheet-metal-simulation.html?view_stride=100&disp_stride=5
```

**Parallel CPU stretch** uses `SharedArrayBuffer` workers when the page is **cross-origin isolated** (`COOP`/`COEP` headers). The capture server sets those headers. When `?webgpu=1` is active, worker-based stretch is disabled so the GPU path owns the constraint solve.

## Capturing displacement images

Displacement maps are written as RGB PNGs (512×512), typically named like `vertex-displacement-rgb-*.png`. Use the in-page capture controls or batch mode with `serve_capture.py` and `scripts/run_sim_batch.py`.

**Cadence:** `view_stride` and `disp_stride` (query parameters on any page that loads the sim, including `batch.html`) control how often view vs displacement files are produced. Headless batch passes them from the CLI:

```bash
python scripts/run_sim_batch.py -n 5 -t 400 --view-stride 100 --disp-stride 5
python scripts/run_sim_batch.py -n 64 -t 300 -j 16
```

Parallel runs use one Chromium process per worker; each worker steps its assigned iterations in order. The capture server uses a threaded HTTP handler so concurrent POSTs from multiple browsers do not block each other.

Place or symlink captures under `data/` for training.

From the **repository root** (parent of `ml/`):

```bash
python ml/train_vae.py --data-dir data --glob "**/vertex-displacement-rgb*.png" --epochs 40 --batch-size 2 --out-dir checkpoints/vae_run1
```

Useful options:

- `--latent-dim` (default 128)
- `--base-ch` (default 48) — channel width in the encoder
- `--device cuda` or `cpu`
- `--lr`, `--image-size` (default 512)

Checkpoints: `checkpoints/.../vae_latest.pt` and per-epoch `vae_epoch_XXXX.pt`. Each checkpoint stores `model`, `latent_dim`, `base_ch`, `image_size`.

## t-SNE latent visualization

After training, generate 2D and/or 3D t-SNE scatter plots of **encoder mean** vectors:

```bash
python scripts/visualize_latent_tsne.py --checkpoint checkpoints/vae_run1/vae_latest.pt --data-dir data --glob "**/vertex-displacement-rgb*.png" --mode both --out-tsne-2d outputs/tsne2d.png --out-tsne-3d outputs/tsne3d.png
```

Options:

- `--max-samples` — random cap for faster embedding (default 800)
- `--perplexity` — t-SNE perplexity (clamped to sample count)
- `--mode 2d` | `3d` | `both`

## Repository layout (relevant pieces)

| Path | Role |
|------|------|
| `simulation/run_sheet_warp.py` | CLI: Warp CUDA mass–spring sheet, displacement PNG export |
| `simulation/cloth_grid.py` | Rectangular grid + springs (from NVIDIA cloth benchmark) |
| `simulation/warp_springs.py` | `eval_springs` + semi-implicit Euler on Warp |
| `simulation/view_3d_mpl.py` | Optional `3dview_*.png` matplotlib renders |
| `scripts/webgpu-cloth-sim.js` | WebGPU cloth kernels (browser-only) |
| `requirements.txt` | `warp-lang`, PyTorch, scikit-learn, matplotlib, etc. |

## Notes

- **WebGPU** requires a compatible browser (recent Chrome/Edge) and may not be available in all automated test environments.
- VAE training on 512×512 images is VRAM-heavy; reduce `--batch-size` on smaller GPUs.
