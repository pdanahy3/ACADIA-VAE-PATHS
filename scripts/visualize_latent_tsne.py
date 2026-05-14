#!/usr/bin/env python3
"""
Encode displacement PNGs with a trained VAE (mean latent) and run t-SNE for 2D / 3D scatter plots.

Run from repository root, for example:

  python scripts/visualize_latent_tsne.py --checkpoint checkpoints/vae/vae_latest.pt \\
    --data-dir data --glob "**/vertex-displacement-rgb*.png" --mode both \\
    --out-tsne-2d outputs/tsne2d.png --out-tsne-3d outputs/tsne3d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from ml.vae import DisplacementVAE  # noqa: E402


def collect_images(data_dir: Path, pattern: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(data_dir.glob(pattern)):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            out.append(p)
    return out


def load_image_tensor(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1)
    return t * 2.0 - 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="t-SNE of VAE latent embeddings for displacement images.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--glob", type=str, default="**/vertex-displacement-rgb*.png")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=800, help="Cap dataset size for t-SNE runtime")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=("2d", "3d", "both"), default="both")
    ap.add_argument("--out-tsne-2d", type=Path, default=Path("outputs/tsne_latent_2d.png"))
    ap.add_argument("--out-tsne-3d", type=Path, default=Path("outputs/tsne_latent_3d.png"))
    args = ap.parse_args()

    paths = collect_images(args.data_dir, args.glob)
    if not paths:
        raise SystemExit(f"No images matched {args.data_dir}/{args.glob}")

    if len(paths) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(paths), size=args.max_samples, replace=False)
        paths = [paths[int(i)] for i in sorted(idx)]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    latent_dim = int(ckpt.get("latent_dim", 128))
    base_ch = int(ckpt.get("base_ch", 48))
    image_size = int(ckpt.get("image_size", args.image_size))

    model = DisplacementVAE(in_channels=3, latent_dim=latent_dim, base=base_ch)
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    latents: list[np.ndarray] = []
    with torch.inference_mode():
        batch: list[torch.Tensor] = []
        for p in tqdm(paths, desc="encode"):
            batch.append(load_image_tensor(p, image_size))
            if len(batch) >= args.batch_size:
                x = torch.stack(batch, dim=0).to(device)
                z = model.encode_mu(x).cpu().numpy()
                latents.append(z)
                batch = []
        if batch:
            x = torch.stack(batch, dim=0).to(device)
            z = model.encode_mu(x).cpu().numpy()
            latents.append(z)

    Z = np.concatenate(latents, axis=0)
    n = Z.shape[0]
    if n < 4:
        raise SystemExit("t-SNE needs at least 4 samples; increase dataset or reduce --max-samples filtering.")
    perp = min(float(args.perplexity), float(n - 1))
    perp = max(2.0, perp)

    colors = np.linspace(0, 1, n)

    args.out_tsne_2d.parent.mkdir(parents=True, exist_ok=True)
    args.out_tsne_3d.parent.mkdir(parents=True, exist_ok=True)
    if args.mode in ("2d", "both"):
        tsne2 = TSNE(n_components=2, perplexity=perp, learning_rate="auto", init="pca", random_state=args.seed)
        Y2 = tsne2.fit_transform(Z)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(Y2[:, 0], Y2[:, 1], c=colors, cmap="viridis", s=12, alpha=0.85)
        ax.set_title("t-SNE of VAE latent (2D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(args.out_tsne_2d, dpi=160)
        plt.close(fig)
        print("Wrote", args.out_tsne_2d)

    if args.mode in ("3d", "both"):
        tsne3 = TSNE(n_components=3, perplexity=perp, learning_rate="auto", init="pca", random_state=args.seed)
        Y3 = tsne3.fit_transform(Z)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Y3[:, 0], Y3[:, 1], Y3[:, 2], c=colors, cmap="viridis", s=10, alpha=0.85)
        ax.set_title("t-SNE of VAE latent (3D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        fig.tight_layout()
        fig.savefig(args.out_tsne_3d, dpi=160)
        plt.close(fig)
        print("Wrote", args.out_tsne_3d)


if __name__ == "__main__":
    main()
