"""Displacement RGB PNG: fixed linear map from rest mesh AABB span to byte channels."""

from __future__ import annotations

import numpy as np
from PIL import Image


def fixed_domain_half_extent_from_rest(rest: np.ndarray) -> float:
    """
    Half-extent M used for all three displacement channels: each component Δ is mapped
    linearly from [-M, M] to [0, 255], where M is the maximum axis span of the **rest**
    vertex bounding box (max of per-axis peak-to-peak over x, y, z).
    """
    r = np.asarray(rest, dtype=np.float64)
    span = np.ptp(r, axis=0)
    m = float(np.max(span)) if span.size else 0.0
    if not np.isfinite(m) or m < 1e-9:
        return 1.0
    return m


def component_to_byte(value: float, half_extent: float) -> int:
    if not np.isfinite(value) or not np.isfinite(half_extent) or half_extent < 1e-12:
        return 127
    t = ((value + half_extent) / (2.0 * half_extent)) * 255.0
    if not np.isfinite(t):
        return 127
    return int(np.clip(round(float(t)), 0, 255))


def displacement_rgb_array(
    positions: np.ndarray,
    rest: np.ndarray,
    nu: int,
    nv: int,
    *,
    d: float | None = None,
) -> np.ndarray:
    """
    (nv, nu, 3) uint8 — R = Δx, G = Δy, B = Δz with the same fixed half-extent M for all
    channels (from rest AABB by default, or ``d`` if given).
    """
    m = fixed_domain_half_extent_from_rest(rest) if d is None else float(d)
    if not np.isfinite(m) or m < 1e-12:
        m = fixed_domain_half_extent_from_rest(rest)
    out = np.zeros((nv, nu, 3), dtype=np.uint8)
    for v in range(nv):
        for u in range(nu):
            i = v * nu + u
            dx, dy, dz = np.nan_to_num(
                (
                    float(positions[i, 0] - rest[i, 0]),
                    float(positions[i, 1] - rest[i, 1]),
                    float(positions[i, 2] - rest[i, 2]),
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            out[v, u, 0] = component_to_byte(dx, m)
            out[v, u, 1] = component_to_byte(dy, m)
            out[v, u, 2] = component_to_byte(dz, m)
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
