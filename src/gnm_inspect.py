from preflight import require

require(gnm=True, packages=("numpy", "matplotlib"))

from gnm.shape import gnm_numpy
from gnm.shape import gnm_landmarks
import numpy as np
import matplotlib.pyplot as plt


# Load GNM V3 HEAD
gnm = gnm_numpy.GNM.from_local(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)


# --------------------------------------------------
# 1. Model dimensions
# --------------------------------------------------

print("=== GNM MODEL ===")

print("Identity dimensions:", gnm.identity_dim)
print("Expression dimensions:", gnm.expression_dim)
print("Number of joints:", gnm.num_joints)


# --------------------------------------------------
# 2. Mesh information
# --------------------------------------------------

template_vertices = gnm.template_vertex_positions
faces = gnm.triangles

print("\n=== MESH ===")

print("Vertices:", template_vertices.shape)
print("Triangles:", faces.shape)


# --------------------------------------------------
# 3. Create the template face
# --------------------------------------------------

identity = np.zeros(gnm.identity_dim)
expression = np.zeros(gnm.expression_dim)
rotations = np.zeros((gnm.num_joints, 3))
translation = np.zeros(3)

vertices = gnm(
    identity=identity,
    expression=expression,
    rotations=rotations,
    translation=translation,
)

print("\nGenerated vertices:", vertices.shape)


# --------------------------------------------------
# 4. Print parameter ranges
# --------------------------------------------------

print("\n=== PARAMETERS ===")

print("Identity shape:", identity.shape)
print("Expression shape:", expression.shape)
print("Rotations shape:", rotations.shape)
print("Translation shape:", translation.shape)


# --------------------------------------------------
# 5. Basic geometry statistics
# --------------------------------------------------

print("\n=== GEOMETRY ===")

print("Min XYZ:", vertices.min(axis=0))
print("Max XYZ:", vertices.max(axis=0))
print("Center:", vertices.mean(axis=0))

# --------------------------------------------------
# 6. GNM landmarks
# --------------------------------------------------

print("\n=== LANDMARKS ===")

vertices, landmarks = gnm.vertices_and_landmarks(
    identity=identity,
    expression=expression,
    rotations=rotations,
    translation=translation,
    landmarks_type=gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68,
)

print("Landmarks type:", gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
print("Landmarks shape:", landmarks.shape)
print("First 5 landmarks:")
print(landmarks[:5])

# --------------------------------------------------
# 7. Visualize GNM landmarks
# --------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    landmarks[:, 0],
    landmarks[:, 1],
    landmarks[:, 2],
    s=10,
)

for i, point in enumerate(landmarks):
    ax.text(
        point[0],
        point[1],
        point[2],
        str(i),
        fontsize=6,
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("GNM HEAD_SPARSE_68 Landmarks")

plt.show()
