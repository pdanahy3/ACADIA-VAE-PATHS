#!/usr/bin/env python3
"""
Train the displacement-map VAE on 512×512 RGB PNGs (e.g. vertex-displacement-rgb-*.png).

Example:
  python ml/train_vae.py --data-dir data --epochs 50 --batch-size 2 --out-dir checkpoints/vae_run1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if __package__ is None:
    sys.path.insert(0, str(_REPO))

if __package__:
    from .vae import DisplacementVAE, vae_loss
else:
    from ml.vae import DisplacementVAE, vae_loss


class DisplacementImageDataset(Dataset):
    def __init__(self, paths: list[Path], size: int = 512) -> None:
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(img)).float() / 255.0
        t = t.permute(2, 0, 1)  # CHW
        t = t * 2.0 - 1.0
        return t


def collect_images(data_dir: Path, pattern: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(data_dir.glob(pattern)):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ConvVAE on displacement RGB images.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder of PNG images")
    ap.add_argument("--glob", type=str, default="**/vertex-displacement-rgb*.png", help="Glob under data-dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--base-ch", type=int, default=48, help="Encoder base channels")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/vae"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = collect_images(args.data_dir, args.glob)
    if not paths:
        raise SystemExit(f"No images matched {args.data_dir}/{args.glob}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta["num_images"] = len(paths)
    (args.out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ds = DisplacementImageDataset(paths, size=args.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = DisplacementVAE(in_channels=3, latent_dim=args.latent_dim, base=args.base_ch).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch.to(device)
            recon, mu, logvar = model(x)
            loss, rloss, kl = vae_loss(recon, x, mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=float(loss), recon=float(rloss), kl=float(kl))
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "latent_dim": args.latent_dim,
            "base_ch": args.base_ch,
            "image_size": args.image_size,
        }
        torch.save(ckpt, args.out_dir / f"vae_epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "latent_dim": args.latent_dim,
            "base_ch": args.base_ch,
            "image_size": args.image_size,
        },
        args.out_dir / "vae_latest.pt",
    )
    print("Saved", args.out_dir / "vae_latest.pt")


if __name__ == "__main__":
    main()
