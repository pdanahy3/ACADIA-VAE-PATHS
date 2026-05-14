"""Off-screen 3D cloth/mesh PNG renders (matplotlib), for run_sheet_warp."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_cloth_3dview_png(
    positions: np.ndarray,
    triangle_indices: np.ndarray,
    path: str | Path,
    *,
    elev_deg: float = 45.0,
    azim_deg: float = 45.0,
    figsize: tuple[float, float] = (9.0, 9.0),
    dpi: int = 120,
) -> None:
    """
    Render triangle mesh with **Z up** (same world axes as the sim), centered on the deformed centroid.
    Camera uses matplotlib ``view_init(elev, azim)`` — defaults 45°, 45° to mirror the browser batch angles.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path = Path(path)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    centroid = np.mean(pos, axis=0)
    pos_c = pos - centroid

    tri = np.asarray(triangle_indices, dtype=np.int64).reshape(-1, 3)
    verts = pos_c[tri]

    fig = plt.figure(figsize=figsize, facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d", facecolor="#111111")

    coll = Poly3DCollection(
        verts,
        facecolors=(0.45, 0.62, 0.88, 0.92),
        edgecolors=(0.15, 0.18, 0.22, 0.45),
        linewidths=0.12,
    )
    ax.add_collection3d(coll)

    span = np.ptp(pos_c, axis=0)
    margin = float(max(span.max() * 0.26, 1.0))
    lo = pos_c.min(axis=0) - margin
    hi = pos_c.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    ax.set_box_aspect(tuple(span + 1e-6))

    ax.view_init(elev=elev_deg, azim=azim_deg)

    ax.set_xlabel("X", color="0.85")
    ax.set_ylabel("Y", color="0.85")
    ax.set_zlabel("Z", color="0.85")
    ax.tick_params(colors="0.75", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.07, 0.07, 0.09, 0.95))
        axis.pane.set_edgecolor((0.28, 0.28, 0.32))
    ax.grid(True, color="0.35", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
