from pathlib import Path

from preflight import require

require(gnm=True, landmark_file=True, packages=("numpy",))

import numpy as np

from gnm.shape import gnm_numpy


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARK_FILE = PROJECT_ROOT / "data" / "head_dense_468_v2_nowarp.txt"


# --------------------------------------------------
# Load correspondence
# --------------------------------------------------

data = np.loadtxt(LANDMARK_FILE)

if data.shape != (468, 6):
    raise ValueError(
        f"Expected (468, 6), got {data.shape}"
    )

indices = data[:, ::2].astype(np.int32)
weights = data[:, 1::2]

print("Correspondence loaded")
print("Indices shape:", indices.shape)
print("Weights shape:", weights.shape)


# --------------------------------------------------
# Load GNM V3 HEAD
# --------------------------------------------------

gnm = gnm_numpy.GNM.from_local(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)


# --------------------------------------------------
# Generate neutral GNM head
# --------------------------------------------------

vertices = np.asarray(
    gnm(None, None, None, None)
)

print("GNM vertices:", vertices.shape)


# --------------------------------------------------
# Evaluate MediaPipe 468 landmarks on GNM
# --------------------------------------------------

landmarks = (
    vertices[indices] * weights[..., None]
).sum(axis=1)

print("GNM MediaPipe landmarks:", landmarks.shape)

print("\nFirst 5 landmarks:")
print(landmarks[:5])


# --------------------------------------------------
# Validate barycentric weights
# --------------------------------------------------

weight_sums = weights.sum(axis=1)

print("\n=== WEIGHT CHECK ===")
print("Min weight sum:", weight_sums.min())
print("Max weight sum:", weight_sums.max())

if not np.allclose(weight_sums, 1.0, atol=1e-5):
    raise ValueError("Some barycentric weights do not sum to 1.")

print("Barycentric weights: OK")


# --------------------------------------------------
# Basic landmark geometry
# --------------------------------------------------

print("\n=== LANDMARK GEOMETRY ===")

print("Min XYZ:", landmarks.min(axis=0))
print("Max XYZ:", landmarks.max(axis=0))
print("Center:", landmarks.mean(axis=0))
