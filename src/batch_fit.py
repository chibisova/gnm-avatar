"""Batch-fit GNM to a folder of photos with two landmark correspondence versions.

For every photo and every version this runs `fit_gnm.py` in a subprocess, then
writes:

    results/<version>/<photo>/        fit outputs + metrics.json + fit.log
    results/metrics.csv               one row per photo and version
    results/metrics.md                same table + v1-vs-v2 comparison
    results/summary_grid.png          mesh overlays, photos x versions

Normalised error (NME) = landmark error in pixels / inter-ocular distance,
measured on the MediaPipe landmarks of the photo (outer eye corners 33 and 263).

    python src/batch_fit.py                    # fit + summarise
    python src/batch_fit.py --summary-only     # re-summarise existing results
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIT_SCRIPT = PROJECT_ROOT / "src" / "fit_gnm.py"

VERSIONS = {
    "v1": PROJECT_ROOT / "data" / "head_dense_468_v1.txt",
    "v2": PROJECT_ROOT / "data" / "head_dense_468_v2.txt",
    "v2-nowarp": PROJECT_ROOT / "data" / "head_dense_468_v2_nowarp.txt",
}

EYE_OUTER_RIGHT = 33
EYE_OUTER_LEFT = 263

# Face-outline points next to the ears, reported separately.
EAR_POINTS = (93, 323, 234, 454)

REGION_COLUMNS = {
    "Right eye": "nme_right_eye",
    "Left eye": "nme_left_eye",
    "Nose": "nme_nose",
    "Mouth": "nme_mouth",
    "Face contour": "nme_contour",
}

CSV_COLUMNS = [
    "photo",
    "version",
    "status",
    "mean_error_px",
    "interocular_px",
    "nme_all",
    *REGION_COLUMNS.values(),
    *(f"nme_lm{i}" for i in EAR_POINTS),
    "identity_at_limit",
    "expression_at_limit",
    "fit_seconds",
]


def run_fits(photos: list[Path], results_dir: Path, versions: dict[str, Path]) -> None:
    total = len(photos) * len(versions)
    done = 0
    for version, landmark_file in versions.items():
        if not landmark_file.is_file():
            sys.exit(f"Correspondence file for {version} not found: {landmark_file}")
        for photo in photos:
            done += 1
            out_dir = results_dir / version / photo.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics.json"
            metrics_path.unlink(missing_ok=True)

            print(f"[{done}/{total}] {version} {photo.name} ...", flush=True)
            start = time.perf_counter()
            with (out_dir / "fit.log").open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(FIT_SCRIPT),
                        "--image", str(photo),
                        "--landmarks", str(landmark_file),
                        "--out-dir", str(out_dir),
                        "--metrics-json", str(metrics_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                )
            wall = time.perf_counter() - start
            status = "ok" if proc.returncode == 0 and metrics_path.is_file() else "failed"
            print(f"    {status} ({wall:.0f} s wall)", flush=True)


def photo_row(photo: str, version: str, metrics_path: Path) -> dict:
    import numpy as np

    if not metrics_path.is_file():
        return {"photo": photo, "version": version, "status": "failed"}

    m = json.loads(metrics_path.read_text())
    errors = np.asarray(m["errors_px"])
    target = np.asarray(m["target_landmarks_px"])
    interocular = float(np.linalg.norm(target[EYE_OUTER_RIGHT] - target[EYE_OUTER_LEFT]))
    nme = errors / interocular

    row = {
        "photo": photo,
        "version": version,
        "status": "ok",
        "mean_error_px": round(float(errors.mean()), 2),
        "interocular_px": round(interocular, 1),
        "nme_all": round(float(nme.mean()), 4),
        "identity_at_limit": m["identity_at_limit"],
        "expression_at_limit": m["expression_at_limit"],
        "fit_seconds": round(m["fit_seconds"], 1),
    }
    for region, column in REGION_COLUMNS.items():
        row[column] = round(float(nme[m["regions"][region]].mean()), 4)
    for i in EAR_POINTS:
        row[f"nme_lm{i}"] = round(float(nme[i]), 4)
    return row


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_summary(rows: list[dict], results_dir: Path, versions: list[str]) -> None:
    with (results_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    ear_cols = [f"nme_lm{i}" for i in EAR_POINTS]
    per_photo = md_table(
        ["Photo", "Version", "Mean err (px)", "NME all", "R eye", "L eye",
         "Nose", "Mouth", "Contour", *(f"lm {i}" for i in EAR_POINTS),
         "Identity at ±3 (/253)", "Expression at ±3 (/383)", "Fit time (s)"],
        [
            [r["photo"], r["version"], r["mean_error_px"], f'{r["nme_all"]:.4f}',
             *(f'{r[c]:.4f}' for c in REGION_COLUMNS.values()),
             *(f'{r[c]:.4f}' for c in ear_cols),
             r["identity_at_limit"], r["expression_at_limit"], r["fit_seconds"]]
            if r["status"] == "ok"
            else [r["photo"], r["version"], "failed", *[""] * (10 + len(ear_cols))]
            for r in rows
        ],
    )

    ok = {v: [r for r in rows if r["version"] == v and r["status"] == "ok"] for v in versions}
    metrics = [("NME all", "nme_all", 4)]
    metrics += [(f"NME {name.lower()}", col, 4) for name, col in REGION_COLUMNS.items()]
    metrics += [(f"NME point {i}", f"nme_lm{i}", 4) for i in EAR_POINTS]
    metrics += [
        ("Identity at ±3 (/253)", "identity_at_limit", 1),
        ("Expression at ±3 (/383)", "expression_at_limit", 1),
        ("Fit time (s)", "fit_seconds", 1),
    ]
    base, others = versions[0], versions[1:]
    compare_rows = []
    for label, col, digits in metrics:
        vals = {v: mean([r[col] for r in ok[v]]) for v in versions}
        changes = [
            f"{(vals[v] - vals[base]) / vals[base] * 100:+.1f}%" if vals[base] else ""
            for v in others
        ]
        compare_rows.append(
            [label, *(f"{vals[v]:.{digits}f}" for v in versions), *changes]
        )
    comparison = md_table(
        ["Mean over photos", *versions, *(f"{v} vs {base}" for v in others)],
        compare_rows,
    )

    n_photos = len({r["photo"] for r in rows})
    text = (
        "# GNM fit metrics\n\n"
        f"{n_photos} photos x {len(versions)} landmark versions. "
        "NME (landmark residual) = landmark error / inter-ocular distance (MediaPipe 33-263), "
        "averaged over the region's landmarks. Fit time = optimisation "
        "stages 1 + 2 only (CPU). Points 93/323 and 234/454 are the face-outline "
        "points next to the ears.\n\n"
        f"## {' vs '.join(versions)}\n\n{comparison}\n\n"
        f"## Per photo\n\n{per_photo}\n"
    )
    # Keep the section written by identity_check.py, if any.
    md_path = results_dir / "metrics.md"
    if md_path.is_file() and "## Identity check" in md_path.read_text():
        text += "\n## Identity check" + md_path.read_text().split("## Identity check", 1)[1]
    md_path.write_text(text)


def face_crop(image, metrics_path: Path, margin: float = 0.6):
    """Square crop around the MediaPipe face landmarks (whole image if unknown)."""
    import numpy as np

    if not metrics_path.is_file():
        return image
    points = np.asarray(json.loads(metrics_path.read_text())["target_landmarks_px"])
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    h, w = image.shape[:2]
    left, top = int(max(0, cx - half)), int(max(0, cy - half))
    right, bottom = int(min(w, cx + half)), int(min(h, cy + half))
    return image[top:bottom, left:right]


def write_grid(photos: list[str], results_dir: Path, versions: list[str],
               rows: list[dict], cell: int = 420) -> Path:
    import cv2
    import numpy as np

    label_h = 44
    by_key = {(r["photo"], r["version"]): r for r in rows}
    grid_rows = []
    for photo in photos:
        cells = []
        for version in versions:
            out_dir = results_dir / version / photo
            image = cv2.imread(str(out_dir / "gnm_fit_mesh_overlay.png"))
            canvas = np.full((cell + label_h, cell, 3), 30, np.uint8)
            if image is not None:
                image = face_crop(image, out_dir / "metrics.json")
                scale = cell / max(image.shape[:2])
                image = cv2.resize(
                    image,
                    (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                y = label_h + (cell - image.shape[0]) // 2
                x = (cell - image.shape[1]) // 2
                canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
            row = by_key.get((photo, version), {})
            line1 = f"{photo}  {version}"
            line2 = "failed"
            if row.get("status") == "ok":
                line2 = (f"NME {row['nme_all']:.4f}  id@3 {row['identity_at_limit']}/253"
                         f"  ex@3 {row['expression_at_limit']}/383")
            for i, text in enumerate((line1, line2)):
                cv2.putText(canvas, text, (8, 18 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(canvas)
        grid_rows.append(np.hstack(cells))

    grid_path = results_dir / "summary_grid.png"
    cv2.imwrite(str(grid_path), np.vstack(grid_rows))
    return grid_path


def write_ear_check(photos: list[str], results_dir: Path, versions: list[str],
                    cell: int = 300) -> Path:
    """Crops around both ears (MediaPipe 234 / 454) for every version, side by side.

    Red dots: MediaPipe points 93, 323, 234, 454 detected in the photo.
    """
    import cv2
    import numpy as np

    rows = []
    for photo in photos:
        cells = []
        for version in versions:
            out_dir = results_dir / version / photo
            image = cv2.imread(str(out_dir / "gnm_fit_mesh_overlay.png"))
            metrics_path = out_dir / "metrics.json"
            for centre in (234, 454):
                canvas = np.full((cell, cell, 3), 30, np.uint8)
                if image is not None and metrics_path.is_file():
                    t = np.asarray(json.loads(metrics_path.read_text())["target_landmarks_px"])
                    half = 0.55 * np.linalg.norm(t[EYE_OUTER_RIGHT] - t[EYE_OUTER_LEFT])
                    cx, cy = t[centre]
                    x0, y0 = int(cx - half), int(cy - half)
                    size = int(2 * half)
                    # Pad so crops near the image border keep the same scale.
                    padded = cv2.copyMakeBorder(image, size, size, size, size,
                                                cv2.BORDER_CONSTANT, value=(30, 30, 30))
                    crop = padded[y0 + size:y0 + 2 * size, x0 + size:x0 + 2 * size]
                    canvas = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
                    for i in EAR_POINTS:
                        x = int((t[i, 0] - x0) / size * cell)
                        y = int((t[i, 1] - y0) / size * cell)
                        if 0 <= x < cell and 0 <= y < cell:
                            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
                cv2.rectangle(canvas, (0, 0), (cell, 22), (0, 0, 0), -1)
                cv2.putText(canvas, f"{photo} {version} lm{centre}", (5, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(canvas)
        rows.append(np.hstack(cells))

    path = results_dir / "ear_check.png"
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default=str(PROJECT_ROOT / "data" / "new"),
                        help="Folder of .jpg photos (default: data/new)")
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results"),
                        help="Output folder (default: results/)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip fitting; rebuild metrics and grid from results/")
    args = parser.parse_args()

    photos = sorted(Path(args.images).glob("*.jpg"))
    if not photos:
        sys.exit(f"No .jpg photos found in {args.images}")
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        run_fits(photos, results_dir, VERSIONS)

    versions = list(VERSIONS)
    rows = [
        photo_row(p.stem, v, results_dir / v / p.stem / "metrics.json")
        for p in photos
        for v in versions
    ]
    write_summary(rows, results_dir, versions)
    grid = write_grid([p.stem for p in photos], results_dir, versions, rows)
    ears = write_ear_check([p.stem for p in photos], results_dir, versions)
    print(f"\nMetrics: {results_dir / 'metrics.csv'}, {results_dir / 'metrics.md'}")
    print(f"Grid:    {grid}")
    print(f"Ears:    {ears}")


if __name__ == "__main__":
    main()
