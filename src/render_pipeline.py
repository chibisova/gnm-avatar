"""Pipeline figure for the README: photo -> landmarks -> correspondence -> fit -> mesh.

Writes docs/pipeline.png (one row of 5 panels with arrows), using one photo's
results as the example (default: results/v2-nowarp/P3). The photo crop is drawn
only into this PNG and never saved on its own. Panel 4 redraws the fitted
mesh over the photo without the eyeballs, teeth and mouth interior.

    python src/render_pipeline.py
    python src/render_pipeline.py --version v2-nowarp --photo P3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import render_3d  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EAR_PINS = (93, 323, 234, 454)
CROP_SIZE = 512
DOT = "#1f77b4"
EAR = "#e8590c"
MESH = "#FFFFFF"         # panel 4 wireframe and silhouette (white)
MESH_PX, OUTLINE_PX = 0.4, 1.0   # line widths in output pixels
DPI = 100
# GNM vertex groups left out of the panel 4 wireframe (inner parts that show
# as dark blobs over the photo): eyeballs, eye sockets, teeth, gums, tongue,
# mouth interior.
HIDDEN_GROUPS = ("eyes", "eye_sockets", "teeth", "gums", "tongue", "mouth_sock")


def square_box(points: np.ndarray, margin: float = 0.3) -> tuple[int, int, int]:
    """Centre x, centre y and half size of a square around `points` plus margin."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) / 2
    half = int((hi - lo).max() * (1 + 2 * margin) / 2)
    return int(centre[0]), int(centre[1]), half


def crop(image: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Square RGB crop of a BGR image, resized to CROP_SIZE. In memory only."""
    cx, cy, half = box
    padded = cv2.copyMakeBorder(image, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    out = cv2.resize(out, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def to_crop(points: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    cx, cy, half = box
    return (points - [cx - half, cy - half]) / (2 * half) * CROP_SIZE


def frame(ax, points_xy: np.ndarray, margin: float) -> None:
    """Square axis limits around projected points plus margin (like the photo crops)."""
    lo, hi = points_xy.min(axis=0), points_xy.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() * (1 + 2 * margin) / 2
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)


def rotate_y(points: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return points @ rot.T


def load_gnm():
    from preflight import require

    require(gnm=True)
    from gnm.shape import gnm_numpy

    return gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)


def project(vertices: np.ndarray, params) -> np.ndarray:
    """Image coordinates with the fit's weak-perspective camera (as in fit_gnm.py)."""
    scale = float(params["camera_scale"])
    translation = np.asarray(params["camera_translation"]).reshape(-1)
    return np.stack([scale * vertices[:, 0] + translation[0],
                     -scale * vertices[:, 1] + translation[1]], axis=1)


def face_wireframe(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray,
                   projected: np.ndarray, image_shape) -> tuple[np.ndarray, list[np.ndarray]]:
    """Wireframe edges and silhouette of the fitted mesh, in photo pixel coordinates.

    Edges: triangles with no vertex in HIDDEN_GROUPS and that face the camera;
    returned as an (E, 2, 2) array of unique segments.
    Silhouette: outer contour(s) of the whole projected mesh.
    """
    keep = face_mask[faces].all(axis=1)
    tri = vertices[faces]
    normal_z = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    keep &= render_3d.outward_sign(vertices, faces) * normal_z > 0
    kept = faces[keep]
    edges = np.sort(np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(projected[f]).astype(np.int32) for f in faces], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = [c[:, 0, :].astype(float) for c in contours if len(c) > 50]
    return projected[edges], outline


def show_image(ax, image):
    ax.imshow(image)
    ax.set_xlim(0, CROP_SIZE)
    ax.set_ylim(CROP_SIZE, 0)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v2-nowarp")
    parser.add_argument("--photo", default="P3", help="Folder name under results/<version>/")
    parser.add_argument("--landmarks", default=None,
                        help="Correspondence file for panel 3 (default: the version's file)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "pipeline.png"))
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / args.version / args.photo
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        sys.exit(f"No metrics.json in {run_dir}; run src/batch_fit.py first.")
    metrics = json.loads(metrics_path.read_text())
    photo = cv2.imread(metrics["image"])
    if photo is None:
        sys.exit(f"Photo not found: {metrics['image']}")
    params = np.load(run_dir / "gnm_fit.npz")

    from gnm_mediapipe_landmarks import load_landmarks, sample

    landmark_file = Path(args.landmarks or metrics["landmarks_file"])
    indices, weights = load_landmarks(landmark_file)
    gnm = load_gnm()
    mean_vertices = np.asarray(gnm(None, None, None, None))
    mean_faces = np.asarray(gnm.triangles, dtype=np.int64)
    names = list(gnm.vertex_group_names)
    groups = np.asarray(gnm.vertex_groups)
    face_mask = ~(groups[[names.index(n) for n in HIDDEN_GROUPS]] > 0).any(axis=0)
    pins = sample(mean_vertices, indices, weights) - mean_vertices.mean(axis=0)

    fit_vertices, fit_faces = render_3d.load_obj(run_dir / "gnm_fit.obj")
    fit_projected = project(fit_vertices, params)

    # One square crop for panels 1, 2 and 4, focused on the face: the fitted head
    # from the top of the skull down to the chin (the neck is cut off).
    landmarks_2d = np.asarray(metrics["target_landmarks_px"])
    head = fit_projected[fit_projected[:, 1] <= landmarks_2d[:, 1].max()]
    box = square_box(np.vstack([landmarks_2d, head]), margin=0.04)
    photo_crop = crop(photo, box)
    landmarks_crop = to_crop(landmarks_2d, box)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), facecolor="white",
                             gridspec_kw={"wspace": 0.35})
    titles = ["Input photo", "MediaPipe landmarks", "Correspondence",
              "Two-stage fit", "GNM Head mesh"]
    captions = ["photo", "468 2D landmarks", "landmark → mesh pins",
                "stage 1: pose, stage 2: identity + expression", "fitted mesh"]

    # 1. photo
    show_image(axes[0], photo_crop)

    # 2. landmarks
    show_image(axes[1], photo_crop)
    axes[1].scatter(*landmarks_crop.T, s=1.5, c=DOT, linewidths=0)

    # 3. correspondence on the GNM mean head (front view, orthographic)
    sign = render_3d.outward_sign(mean_vertices, mean_faces)
    render_3d.render_view(axes[2], mean_vertices,
                          render_3d.trim_neck_rim(mean_faces, 8), sign, 0.0)
    others = np.setdiff1d(np.arange(len(pins)), EAR_PINS)
    axes[2].scatter(*pins[others, :2].T, s=1.5, c=DOT, linewidths=0, zorder=3)
    axes[2].scatter(*pins[list(EAR_PINS), :2].T, s=22, c=EAR, edgecolors="white",
                    linewidths=0.6, zorder=4)
    frame(axes[2], pins[:, :2], margin=0.3)
    axes[2].text(0.5, -0.16, "● ear-side pins (see findings)", color=EAR, fontsize=9,
                 ha="center", va="top", transform=axes[2].transAxes)

    # 4. fitted wireframe over the photo: whole mesh minus eyes/teeth/mouth interior
    segments, outline = face_wireframe(fit_vertices, fit_faces, face_mask,
                                       fit_projected, photo.shape)
    show_image(axes[3], photo_crop)
    px = 72 / DPI  # points per output pixel
    axes[3].add_collection(LineCollection(
        to_crop(segments.reshape(-1, 2), box).reshape(-1, 2, 2),
        colors=MESH, linewidths=MESH_PX * px, alpha=0.5, capstyle="round"))
    for contour in outline:
        closed = to_crop(np.vstack([contour, contour[:1]]), box)
        axes[3].plot(*closed.T, color=MESH, linewidth=OUTLINE_PX * px, solid_joinstyle="round")

    # 5. fitted mesh, 3/4 view
    sign = render_3d.outward_sign(fit_vertices, fit_faces)
    render_3d.render_view(axes[4], fit_vertices,
                          render_3d.trim_neck_rim(fit_faces, 8), sign, 45.0)
    fit_pins = sample(fit_vertices, indices, weights) - fit_vertices.mean(axis=0)
    frame(axes[4], rotate_y(fit_pins, 45.0)[:, :2], margin=0.45)

    for ax, title, caption in zip(axes, titles, captions):
        ax.set_title(title, fontsize=13, pad=8)
        ax.text(0.5, -0.06, caption, fontsize=10, color="0.35", ha="center", va="top",
                transform=ax.transAxes)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.17)
    for left, right in zip(axes[:-1], axes[1:]):
        a, b = left.get_position(), right.get_position()
        y = (a.y0 + a.y1) / 2
        fig.add_artist(FancyArrowPatch((a.x1 + 0.006, y), (b.x0 - 0.006, y),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=18, color="0.3", linewidth=1.6))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"Pipeline figure: {out}")


if __name__ == "__main__":
    main()
