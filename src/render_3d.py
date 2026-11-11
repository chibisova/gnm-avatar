"""Render fitted GNM meshes alone, grey-shaded, from front, 3/4 and side.

Reads results/<version>/<photo>/gnm_fit.obj for every photo and writes
results/<version>/summary_grid_3d.png (one row per photo, columns 0°, 45°, 90°).
The first row is the GNM mean head (identity = 0, expression = 0) for reference.
The first column is a square face crop of the input photo (from the path and the
MediaPipe landmarks in <photo>/metrics.json). The crop is drawn only into the
composite PNG and never saved on its own.

Software rendering with matplotlib only (no OpenGL): orthographic view,
back faces removed, faces drawn back to front, flat Lambert shading with a
key and a fill light.

    python src/render_3d.py --version v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIEWS = {"front": 0.0, "3/4": 45.0, "side": 90.0}  # rotation about the vertical (y) axis
KEY_LIGHT = np.array([0.35, 0.45, 1.0])
KEY_LIGHT = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
# Fill light from the opposite side, so faces turned away from the key light
# (e.g. the back of the neck in the side view) do not render near-black.
FILL_LIGHT = np.array([-0.8, 0.1, 0.4])
FILL_LIGHT = FILL_LIGHT / np.linalg.norm(FILL_LIGHT)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def outward_sign(vertices: np.ndarray, faces: np.ndarray) -> float:
    """+1 if triangle winding gives outward normals, else -1."""
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = tri.mean(axis=1) - vertices.mean(axis=0)
    return 1.0 if np.sum(np.einsum("ij,ij->i", normals, outward) > 0) >= len(faces) / 2 else -1.0


def trim_neck_rim(faces: np.ndarray, rings: int) -> np.ndarray:
    """Drop the faces within `rings` edge rings of the open neck edge.

    The neck edge is the longest boundary loop of the mesh. Its lower rim curls
    and, in side views, shows as a detached strip below the neck. Selection is
    by topology only, so the same faces are dropped on every GNM mesh.
    """
    if rings <= 0:
        return faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]

    # Group boundary edges into loops (union-find) and keep the longest one.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in boundary:
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in boundary[:, 0]])
    neck_root = np.bincount(roots).argmax()
    selected = np.zeros(faces.max() + 1, dtype=bool)
    selected[boundary[roots == neck_root].ravel()] = True

    for _ in range(rings):
        touching = selected[faces].any(axis=1)
        selected[faces[touching].ravel()] = True
    return faces[~selected[faces].any(axis=1)]


def render_view(ax, vertices, faces, sign, angle_deg):
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    v = (vertices - vertices.mean(axis=0)) @ rot.T

    tri = v[faces]
    normals = sign * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    keep = (normals[:, 2] > 0) & (length > 0)  # camera looks along -z
    tri, normals = tri[keep], normals[keep] / length[keep, None]

    shade = (0.30 + 0.62 * np.clip(normals @ KEY_LIGHT, 0, 1)
             + 0.25 * np.clip(normals @ FILL_LIGHT, 0, 1))
    shade = np.clip(shade, 0, 1)
    order = np.argsort(tri[:, :, 2].mean(axis=1))  # far first
    colors = np.repeat(shade[order, None], 3, axis=1)

    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=colors,
                                     edgecolors=colors, linewidths=0.15))
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")


def mean_head() -> tuple[np.ndarray, np.ndarray]:
    """GNM mean head (identity = 0, expression = 0, pose = 0) and its triangles."""
    from preflight import require

    require(gnm=True)
    from gnm.shape import gnm_numpy

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    return np.asarray(gnm(None, None, None, None)), np.asarray(gnm.triangles, dtype=np.int64)


def face_crop(metrics_path: Path, margin: float = 0.3) -> np.ndarray | None:
    """Square RGB crop: MediaPipe face box plus `margin` on each side. In memory only."""
    import json

    import cv2

    if not metrics_path.is_file():
        return None
    m = json.loads(metrics_path.read_text())
    image = cv2.imread(m["image"])
    if image is None:
        return None
    pts = np.asarray(m["target_landmarks_px"])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    half = int((hi - lo).max() * (1 + 2 * margin) / 2)
    pad = half
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cx, cy = int(centre[0]) + pad, int(centre[1]) + pad
    crop = padded[cy - half:cy + half, cx - half:cx + half]
    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="Folder name under results/, e.g. v2")
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--neck-rings", type=int, default=8,
                        help="Edge rings to trim at the open neck edge (0 = none)")
    parser.add_argument("--no-photo", action="store_true",
                        help="Skip the input photo column")
    parser.add_argument("--no-mean-head", action="store_true",
                        help="Skip the GNM mean head reference row")
    args = parser.parse_args()

    version_dir = Path(args.results) / args.version
    objs = sorted(version_dir.glob("*/gnm_fit.obj"))
    if not objs:
        sys.exit(f"No fitted meshes found in {version_dir}/*/gnm_fit.obj")

    rows = [] if args.no_mean_head else [("GNM\nmean head", None)]
    rows += [(obj.parent.name, obj) for obj in objs]
    titles = ([] if args.no_photo else ["photo"]) + list(VIEWS)
    first_view = 0 if args.no_photo else 1

    fig, axes = plt.subplots(len(rows), len(titles), figsize=(3 * len(titles), 3 * len(rows)),
                             facecolor="white", squeeze=False)
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12)
    for row, (label, obj) in enumerate(rows):
        axes[row, 0].text(-0.08, 0.5, label, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=12)
        if not args.no_photo:
            ax = axes[row, 0]
            crop = None if obj is None else face_crop(obj.parent / "metrics.json")
            if crop is not None:
                ax.imshow(crop)
            else:
                ax.text(0.5, 0.5, "mean head\n(no photo)" if obj is None else "photo\nnot found",
                        ha="center", va="center", fontsize=10, color="0.5",
                        transform=ax.transAxes)
            ax.set_aspect("equal")
            ax.axis("off")
        vertices, faces = mean_head() if obj is None else load_obj(obj)
        sign = outward_sign(vertices, faces)
        faces = trim_neck_rim(faces, args.neck_rings)
        for col, angle in enumerate(VIEWS.values(), start=first_view):
            render_view(axes[row, col], vertices, faces, sign, angle)
        print(f"rendered {label.replace(chr(10), ' ')}", flush=True)

    fig.tight_layout()
    out = version_dir / "summary_grid_3d.png"
    fig.savefig(out, dpi=110)
    print(f"Grid 3D: {out}")


if __name__ == "__main__":
    main()
