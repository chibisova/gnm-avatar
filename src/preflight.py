"""Startup check for third-party files that are not part of this repository.

The GNM Head model (Google) and the MediaPipe Face Landmarker model must be
downloaded separately. Each script calls `require(...)` before importing them,
so a missing file gives a short message instead of a stack trace.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
LANDMARK_FILE = PROJECT_ROOT / "data" / "head_dense_468_v2_nowarp.txt"

# GNM Head v3.0 weights, relative to a GNM model directory.
GNM_HEAD_NPZ = Path("v3_0") / "gnm_head.npz"

README_HINT = (
    'See the README section "Download required files" for setup steps.'
)


def _find_gnm_package() -> Path | None:
    """Directory of the installed `gnm.shape` package, or None."""
    try:
        spec = importlib.util.find_spec("gnm.shape")
    except ModuleNotFoundError:
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def _check_python_packages(names: list[str]) -> list[str]:
    problems = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            problems.append(
                f"Python package '{name}' is not installed "
                "(run: pip install -r requirements.txt)."
            )
    return problems


def gnm_cache_dir() -> Path:
    """Same lookup order as GNM's `get_default_gnm_cache_dir()`."""
    if env_cache := os.getenv("GNM_CACHE_DIR"):
        return Path(env_cache)
    if xdg_cache := os.getenv("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "gnm" / "models"
    return Path.home() / ".cache" / "gnm" / "models"


def gnm_weight_candidates(package_dir: Path) -> list[Path]:
    """Places where `GNM.from_local()` can find gnm_head.npz.

    - GNM cache directory: current GNM releases, where `from_local()`
      reads `<cache>/v3_0/gnm_head.npz`.
    - Inside the package: older GNM clones (before 2026-09-29) shipped
      the weights at `gnm/shape/data/versions/v3_0/gnm_head.npz`.
    """
    return [
        gnm_cache_dir() / GNM_HEAD_NPZ,
        package_dir / "data" / "versions" / GNM_HEAD_NPZ,
    ]


def _check_gnm() -> list[str]:
    package_dir = _find_gnm_package()
    if package_dir is None:
        return [
            "GNM Head is not installed (Python package 'gnm.shape' not found)."
        ]
    candidates = gnm_weight_candidates(package_dir)
    if any(path.is_file() for path in candidates):
        return []
    searched = "\n".join(f"      {path}" for path in candidates)
    return [f"GNM Head model weights (gnm_head.npz) not found. Looked in:\n{searched}"]


def _check_file(path: Path, label: str) -> list[str]:
    if path.is_file():
        return []
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = path
    return [f"{label} not found: {shown}"]


def require(
    gnm: bool = False,
    face_model: bool = False,
    landmark_file: bool = False,
    packages: tuple[str, ...] = (),
) -> None:
    """Exit with a clear message if any required file or package is missing."""
    problems = _check_python_packages(list(packages))
    if gnm:
        problems += _check_gnm()
    if face_model:
        problems += _check_file(FACE_MODEL, "MediaPipe Face Landmarker model")
    if landmark_file:
        problems += _check_file(LANDMARK_FILE, "MediaPipe -> GNM correspondence")

    if not problems:
        return

    print("Missing required files:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print(f"\n{README_HINT}", file=sys.stderr)
    sys.exit(1)
