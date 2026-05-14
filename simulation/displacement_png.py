"""Displacement RGB PNG compatible with scripts/sheet-metal-simulation.js encoding."""

from __future__ import annotations

import numpy as np
from PIL import Image


def displacement_domain_half_extent(positions: np.ndarray, rest: np.ndarray) -> float:
    disp = positions - rest
    sx = float(np.max(disp[:, 0]) - np.min(disp[:, 0]))
    sy = float(np.max(disp[:, 1]) - np.min(disp[:, 1]))
    sz = float(np.max(disp[:, 2]) - np.min(disp[:, 2]))
    return max(sx, sy, sz, 1e-9)


def component_to_byte(value: float, d: float) -> int:
    if d < 1e-12:
        return 127
    t = ((value + d) / (2.0 * d)) * 255.0
    return int(np.clip(round(t), 0, 255))


def displacement_rgb_array(
    positions: np.ndarray,
    rest: np.ndarray,
    nu: int,
    nv: int,
    *,
    d: float | None = None,
) -> np.ndarray:
    """
    (nv, nu, 3) uint8 — same channel mapping as buildDisplacementMapBlob:
    R = Δx, G = Δz, B = Δy (Three.js-style mapping used in the browser export).
    """
    if d is None:
        d = displacement_domain_half_extent(positions, rest)
    out = np.zeros((nv, nu, 3), dtype=np.uint8)
    for v in range(nv):
        for u in range(nu):
            i = v * nu + u
            dx = float(positions[i, 0] - rest[i, 0])
            dy = float(positions[i, 1] - rest[i, 1])
            dz = float(positions[i, 2] - rest[i, 2])
            d_x = dx
            d_y = dz
            d_z = dy
            out[v, u, 0] = component_to_byte(d_x, d)
            out[v, u, 1] = component_to_byte(d_y, d)
            out[v, u, 2] = component_to_byte(d_z, d)
    return out


def save_displacement_png(
    path: str,
    positions: np.ndarray,
    rest: np.ndarray,
    nu: int,
    nv: int,
    *,
    out_size: int | None = 512,
    d: float | None = None,
) -> None:
    rgb = displacement_rgb_array(positions, rest, nu, nv, d=d)
    img = Image.fromarray(rgb, mode="RGB")
    if out_size is not None and (nu != out_size or nv != out_size):
        img = img.resize((out_size, out_size), Image.Resampling.LANCZOS)
    img.save(path)
