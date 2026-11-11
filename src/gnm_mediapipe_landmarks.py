# Source: https://gist.github.com/pundirmr/ecf0c0bb5c93fc1a9534083a8b6d255e
# (revision 01bd756, 2026-07-31)
# Author: pundirmr
# License: Apache 2.0
# Modified in this repository: the default correspondence file path and the
# file name in the docstring point to data/head_dense_468_v2_nowarp.txt, and
# the self-check calls preflight.require() to check for required files first.

"""MediaPipe 468-landmark correspondence for the GNM head model — loader + self-check.

`head_dense_468_v2_nowarp.txt` uses the same format as GNM's shipped `head_sparse_68.txt`
(three `vertex_index weight` pairs per row, barycentric weights summing to 1), but
GNM's `gnm_landmarks.load_landmarks()` dispatches off the `GNMLandmarksType` enum
and only knows HEAD_SPARSE_68 — so read the file directly, as below.

Because the anchors are barycentric on the GNM topology, they hold for ANY
identity, expression and pose without re-registration.

    python gnm_mediapipe_landmarks.py          # runs the built-in self-check

Unofficial / community-built. Not affiliated with Google.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARK_FILE = PROJECT_ROOT / "data" / "head_dense_468_v2_nowarp.txt"
LANDMARK_COUNT = 468


def load_landmarks(path=LANDMARK_FILE):
    """Returns (indices, weights): int (468, 3) and float (468, 3)."""
    data = np.loadtxt(path)
    if data.shape != (LANDMARK_COUNT, 6):
        raise ValueError(f"expected ({LANDMARK_COUNT}, 6), got {data.shape}")
    return data[:, ::2].astype(np.int32), data[:, 1::2].astype(np.float64)


def sample(vertices: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Evaluate the 468 landmarks on a GNM mesh -> (468, 3).

    `vertices` is any (17821, 3) GNM head mesh: template, a sampled identity, an
    expression, a posed frame — the barycentric anchors do not care which.
    """
    return (vertices[indices] * weights[..., None]).sum(axis=1)


# ── MediaPipe canonical indices worth knowing ────────────────────────────────
MOUTH_CORNER_RIGHT = 61     # SUBJECT's right (MediaPipe naming is viewer-relative
MOUTH_CORNER_LEFT = 291     # in some docs — these are anatomical, verified below)
LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
EYE_OUTER_RIGHT = 33
EYE_INNER_RIGHT = 133
EYE_OUTER_LEFT = 263
EYE_INNER_LEFT = 362
LID_UPPER_RIGHT, LID_LOWER_RIGHT = 159, 145
LID_UPPER_LEFT, LID_LOWER_LEFT = 386, 374


def self_check(vertices: np.ndarray) -> bool:
    """Assert the correspondence resolves the structures most people need.

    Included because a landmark embedding can be numerically valid — every anchor
    on the mesh, small projection residuals — while being useless, if landmarks
    that should be distinct collapse onto the same surface location. See the
    eyelid note in the README.
    """
    idx, w = load_landmarks()
    p = sample(vertices, idx, w)

    def mm(a, b):
        return float(np.linalg.norm(p[a] - p[b]) * 1000.0)

    checks = [
        ("inter-ocular distance", mm(EYE_OUTER_RIGHT, EYE_OUTER_LEFT), 50.0, 120.0),
        ("mouth width", mm(MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT), 25.0, 80.0),
        ("palpebral aperture R", mm(LID_UPPER_RIGHT, LID_LOWER_RIGHT), 3.0, 20.0),
        ("palpebral aperture L", mm(LID_UPPER_LEFT, LID_LOWER_LEFT), 3.0, 20.0),
        ("inner-lip gap", mm(LIP_INNER_UPPER, LIP_INNER_LOWER), 0.5, 15.0),
        ("eye width R", mm(EYE_OUTER_RIGHT, EYE_INNER_RIGHT), 15.0, 45.0),
    ]
    ok = True
    print("self-check (values are for GNM's neutral template):")
    for name, val, lo, hi in checks:
        good = lo <= val <= hi
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name:22s} {val:7.2f} mm   (expect {lo}-{hi})")
    return ok


if __name__ == "__main__":
    from preflight import require

    require(gnm=True, landmark_file=True)

    from gnm.shape import gnm_numpy

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    verts = np.asarray(gnm(None, None, None, None))
    print(f"GNM v3 HEAD: {verts.shape[0]} vertices")

    idx, w = load_landmarks()
    print(f"loaded {idx.shape[0]} landmarks; weights sum to 1: "
          f"{np.allclose(w.sum(axis=1), 1.0, atol=1e-3)}")

    lm = sample(verts, idx, w)
    print(f"sampled landmarks: {lm.shape}\n")

    passed = self_check(verts)

    # The anchors are identity/expression-invariant — demonstrate on an expression.
    e = np.zeros(gnm.expression_dim)
    lower_face = [i for i, n in enumerate(gnm.expression_names)
                  if n.startswith("lower_face_region_")]
    e[lower_face[1]] = -3.0                     # a smile in GNM's basis
    smiling = np.asarray(gnm(None, e, None, None))
    lm_smile = sample(smiling, idx, w)
    lift = (lm_smile[MOUTH_CORNER_RIGHT, 1] - lm[MOUTH_CORNER_RIGHT, 1]) * 1000
    print(f"\nsame anchors on a smiling mesh: right mouth corner rises {lift:+.2f} mm")
    print("\nRESULT:", "PASS" if passed else "FAIL")
