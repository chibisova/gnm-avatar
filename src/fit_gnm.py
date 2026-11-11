from pathlib import Path
import argparse
import json
import time

from preflight import require

require(
    gnm=True,
    face_model=True,
    landmark_file=True,
    packages=("cv2", "numpy", "torch", "mediapipe", "trimesh"),
)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import trimesh

from gnm.shape import gnm_pytorch


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANDMARK_FILE = PROJECT_ROOT / "data" / "head_dense_468_v2_nowarp.txt"
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"


# ============================================================
# Configuration
# ============================================================

NUM_LANDMARKS = 468

STAGE_1_STEPS = 1500
STAGE_2_STEPS = 1500

LR_STAGE_1 = 0.03
LR_STAGE_2 = 0.01

DEVICE = "cpu"

LANDMARK_RADIUS = 5
ERROR_LINE_THICKNESS = 1
MESH_LINE_THICKNESS = 1
MESH_LINE_COLOR = (40, 40, 40)  # BGR, dark grey, drawn at full opacity


# ============================================================
# Landmark groups
# ============================================================

RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

LEFT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466
]

NOSE = [
    1, 2, 4, 5, 6, 19, 94, 168, 195, 197,
    98, 327, 326, 97, 278, 344, 440, 275
]

MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321,
    375, 291, 409, 270, 269, 267, 0, 37, 39,
    40, 185, 146, 76, 95, 191, 80, 81, 82, 13,
    14, 312, 311, 310, 415, 308, 324, 318
]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


REGIONS = {
    "Right eye": RIGHT_EYE,
    "Left eye": LEFT_EYE,
    "Nose": NOSE,
    "Mouth": MOUTH,
    "Face contour": FACE_OVAL,
}


# ============================================================
# Landmark weights
# ============================================================

def create_landmark_weights(device):
    weights = torch.ones(
        NUM_LANDMARKS,
        dtype=torch.float32,
        device=device,
    )

    weights[RIGHT_EYE] = 5.0
    weights[LEFT_EYE] = 5.0

    weights[NOSE] = 2.0
    weights[MOUTH] = 2.0

    weights[FACE_OVAL] = 1.0

    return weights


# ============================================================
# Weighted landmark loss
# ============================================================

def calculate_landmark_loss(
    projected,
    target_landmarks,
    landmark_weights,
):
    per_landmark_loss = F.smooth_l1_loss(
        projected,
        target_landmarks,
        reduction="none",
    )

    per_landmark_loss = per_landmark_loss.mean(
        dim=-1
    )

    weighted_loss = (
        per_landmark_loss
        * landmark_weights.unsqueeze(0)
    )

    return weighted_loss.mean()


# ============================================================
# MediaPipe
# ============================================================

def detect_mediapipe_landmarks(image):
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(FACE_MODEL)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError(
            "MediaPipe did not detect a face."
        )

    landmarks = result.face_landmarks[0]

    if len(landmarks) < NUM_LANDMARKS:
        raise RuntimeError(
            f"Expected at least {NUM_LANDMARKS} landmarks, "
            f"got {len(landmarks)}"
        )

    height, width = image.shape[:2]

    points = np.zeros(
        (NUM_LANDMARKS, 2),
        dtype=np.float32,
    )

    for i in range(NUM_LANDMARKS):
        points[i, 0] = landmarks[i].x * width
        points[i, 1] = landmarks[i].y * height

    return points


# ============================================================
# MediaPipe -> GNM correspondence
# ============================================================

def load_correspondence():
    data = np.loadtxt(
        LANDMARK_FILE
    )

    if data.shape != (468, 6):
        raise ValueError(
            f"Expected correspondence shape (468, 6), "
            f"got {data.shape}"
        )

    indices = data[:, ::2].astype(
        np.int64
    )

    weights = data[:, 1::2].astype(
        np.float32
    )

    weight_sums = weights.sum(
        axis=1
    )

    if not np.allclose(
        weight_sums,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Barycentric weights do not sum to 1."
        )

    return indices, weights


# ============================================================
# GNM landmark extraction
# ============================================================

def extract_correspondence(
    vertices,
    indices,
    weights,
):
    landmark_indices = torch.tensor(
        indices,
        dtype=torch.long,
        device=vertices.device,
    )

    landmark_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=vertices.device,
    )

    sampled_vertices = vertices[
        :,
        landmark_indices,
        :
    ]

    landmark_weights = (
        landmark_weights
        .unsqueeze(0)
        .unsqueeze(-1)
    )

    landmarks = (
        sampled_vertices
        * landmark_weights
    ).sum(dim=2)

    return landmarks


# ============================================================
# Projection
# ============================================================

def project_landmarks(
    landmarks_3d,
    scale,
    translation,
):
    x = landmarks_3d[..., 0]
    y = landmarks_3d[..., 1]

    projected_x = (
        scale * x
        + translation[..., 0]
    )

    projected_y = (
        -scale * y
        + translation[..., 1]
    )

    return torch.stack(
        [
            projected_x,
            projected_y,
        ],
        dim=-1,
    )


# ============================================================
# Camera initialization
# ============================================================

def initialize_camera(
    target_landmarks,
    model_landmarks,
):
    target_min = target_landmarks.min(
        axis=0
    )

    target_max = target_landmarks.max(
        axis=0
    )

    model_min = model_landmarks.min(
        axis=0
    )

    model_max = model_landmarks.max(
        axis=0
    )

    target_width = (
        target_max[0]
        - target_min[0]
    )

    target_height = (
        target_max[1]
        - target_min[1]
    )

    model_width = (
        model_max[0]
        - model_min[0]
    )

    model_height = (
        model_max[1]
        - model_min[1]
    )

    scale_x = (
        target_width
        / model_width
    )

    scale_y = (
        target_height
        / model_height
    )

    scale = 0.5 * (
        scale_x + scale_y
    )

    target_center = (
        target_landmarks.mean(axis=0)
    )

    model_center = (
        model_landmarks.mean(axis=0)
    )

    translation_x = (
        target_center[0]
        - scale * model_center[0]
    )

    translation_y = (
        target_center[1]
        + scale * model_center[1]
    )

    return (
        scale,
        translation_x,
        translation_y,
    )


# ============================================================
# Error statistics
# ============================================================

def calculate_errors(
    target,
    predicted,
):
    difference = (
        predicted - target
    )

    errors = np.linalg.norm(
        difference,
        axis=1,
    )

    return errors


def print_error_statistics(
    target,
    predicted,
):
    errors = calculate_errors(
        target,
        predicted,
    )

    print("\n=== GLOBAL ERROR ===")

    print(
        f"Mean error:   {errors.mean():.2f} px"
    )

    print(
        f"Median error: {np.median(errors):.2f} px"
    )

    print(
        f"RMSE:         "
        f"{np.sqrt(np.mean(errors ** 2)):.2f} px"
    )

    print(
        f"P95 error:    "
        f"{np.percentile(errors, 95):.2f} px"
    )

    print(
        f"Max error:    {errors.max():.2f} px"
    )

    print("\n=== REGIONAL ERROR ===")

    for name, region_indices in REGIONS.items():

        region_errors = errors[
            region_indices
        ]

        print(
            f"{name:15s} | "
            f"mean {region_errors.mean():6.2f} px | "
            f"median {np.median(region_errors):6.2f} px | "
            f"p95 {np.percentile(region_errors, 95):6.2f} px"
        )

    return errors


# ============================================================
# Diagnostic landmark image
# ============================================================

def create_diagnostic_image(
    image,
    target,
    predicted,
    errors,
):
    output = image.copy()

    for i in range(NUM_LANDMARKS):

        target_point = (
            int(target[i, 0]),
            int(target[i, 1]),
        )

        predicted_point = (
            int(predicted[i, 0]),
            int(predicted[i, 1]),
        )

        cv2.line(
            output,
            target_point,
            predicted_point,
            (255, 255, 0),
            ERROR_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    for x, y in target:

        cv2.circle(
            output,
            (int(x), int(y)),
            LANDMARK_RADIUS,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    for x, y in predicted:

        cv2.circle(
            output,
            (int(x), int(y)),
            LANDMARK_RADIUS,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

    important = {
        1: "nose",
        33: "R eye",
        61: "R mouth",
        263: "L eye",
        291: "L mouth",
    }

    for index, label in important.items():

        x = int(target[index, 0])
        y = int(target[index, 1])

        cv2.putText(
            output,
            f"{index} {label}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        output,
        (20, 20),
        (550, 145),
        (0, 0, 0),
        -1,
    )

    legend = [
        "GREEN = MediaPipe",
        "RED = fitted GNM",
        "CYAN = error vector",
        f"Mean: {errors.mean():.2f}px",
        f"Median: {np.median(errors):.2f}px",
        f"P95: {np.percentile(errors, 95):.2f}px",
    ]

    for i, text in enumerate(legend):

        cv2.putText(
            output,
            text,
            (35, 45 + i * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# Project full GNM mesh onto image
# ============================================================

def project_mesh_to_image(
    vertices,
    scale,
    camera_translation,
):
    """
    Project all GNM vertices using the same weak-perspective
    camera used for the landmark fitting.

    vertices:
        [V, 3]

    returns:
        [V, 2]
    """

    vertices = np.asarray(
        vertices,
        dtype=np.float32,
    )

    x = vertices[:, 0]
    y = vertices[:, 1]

    projected_x = (
        scale * x
        + camera_translation[0]
    )

    projected_y = (
        -scale * y
        + camera_translation[1]
    )

    return np.stack(
        [
            projected_x,
            projected_y,
        ],
        axis=1,
    )


# ============================================================
# Mesh overlay
# ============================================================

def create_mesh_overlay(
    image,
    vertices,
    triangles,
    scale,
    camera_translation,
):
    """
    Draw the complete GNM triangle topology on top of the
    original photograph.
    """

    projected = project_mesh_to_image(
        vertices,
        scale,
        camera_translation,
    )

    height, width = image.shape[:2]

    # Draw mesh triangles as thin dark lines at full opacity.

    mesh_layer = image.copy()

    for triangle in triangles:

        p0 = projected[triangle[0]]
        p1 = projected[triangle[1]]
        p2 = projected[triangle[2]]

        points = np.array(
            [
                p0,
                p1,
                p2,
            ],
            dtype=np.int32,
        )

        # Skip triangles completely outside the image.
        if (
            points[:, 0].max() < 0
            or points[:, 0].min() >= width
            or points[:, 1].max() < 0
            or points[:, 1].min() >= height
        ):
            continue

        cv2.polylines(
            mesh_layer,
            [points],
            True,
            MESH_LINE_COLOR,
            MESH_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    return mesh_layer


# ============================================================
# Export GNM mesh
# ============================================================

def export_gnm_mesh(
    vertices,
    triangles,
    output_path,
):
    """
    Export fitted GNM mesh as OBJ.
    """

    vertices = np.asarray(
        vertices,
        dtype=np.float32,
    )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        process=False,
    )

    mesh.export(
        str(output_path),
        file_type="obj",
    )

    return mesh


# ============================================================
# Main
# ============================================================

def main():

    global LANDMARK_FILE

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input face image",
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="Open interactive 3D GNM viewer",
    )

    parser.add_argument(
        "--landmarks",
        default=str(LANDMARK_FILE),
        help="MediaPipe -> GNM correspondence file "
        "(default: data/head_dense_468_v2_nowarp.txt)",
    )

    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Folder for the output files (default: data/)",
    )

    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path: also write fit metrics as JSON",
    )

    args = parser.parse_args()

    # Correspondence file used by load_correspondence().
    LANDMARK_FILE = Path(args.landmarks)

    if not LANDMARK_FILE.is_file():
        raise SystemExit(
            f"Correspondence file not found: {LANDMARK_FILE}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    # ========================================================
    # Load image
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    print(
        "Image:",
        image_path,
    )

    print(
        "Resolution:",
        width,
        "x",
        height,
    )

    # ========================================================
    # MediaPipe
    # ========================================================

    print("\n=== MEDIAPIPE ===")

    target_landmarks_np = (
        detect_mediapipe_landmarks(
            image
        )
    )

    print(
        "Detected landmarks:",
        target_landmarks_np.shape,
    )

    # ========================================================
    # Correspondence
    # ========================================================

    print("\n=== CORRESPONDENCE ===")

    indices, weights = (
        load_correspondence()
    )

    print(
        "Indices:",
        indices.shape,
    )

    print(
        "Weights:",
        weights.shape,
    )

    # ========================================================
    # GNM
    # ========================================================

    print("\n=== GNM ===")

    gnm = gnm_pytorch.GNM.from_local(
        version=gnm_pytorch.GNMMajorVersion.V3,
        variant=gnm_pytorch.GNMVariant.HEAD,
    )

    gnm = gnm.to(DEVICE)

    print(
        "Identity dimensions:",
        gnm.identity_dim,
    )

    print(
        "Expression dimensions:",
        gnm.expression_dim,
    )

    print(
        "Joints:",
        gnm.num_joints,
    )

    print(
        "Vertices:",
        gnm.num_vertices,
    )

    print(
        "Triangles:",
        gnm.triangles.shape[0],
    )

    print(
        "Joint names:",
        list(gnm.joint_names),
    )

    # ========================================================
    # GNM parameters
    # ========================================================

    identity = torch.zeros(
        (1, gnm.identity_dim),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    expression = torch.zeros(
        (1, gnm.expression_dim),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    rotations = torch.zeros(
        (1, gnm.num_joints, 3),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    model_translation = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    # ========================================================
    # Initial mesh
    # ========================================================

    with torch.no_grad():

        vertices = gnm(
            identity,
            expression,
            rotations,
            model_translation,
        )

        landmarks_3d = (
            extract_correspondence(
                vertices,
                indices,
                weights,
            )
        )

    model_landmarks_np = (
        landmarks_3d[0]
        .cpu()
        .numpy()
    )

    # ========================================================
    # Initial camera
    # ========================================================

    (
        scale_init,
        tx_init,
        ty_init,
    ) = initialize_camera(
        target_landmarks_np,
        model_landmarks_np,
    )

    scale = torch.tensor(
        scale_init,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    camera_translation = torch.tensor(
        [[tx_init, ty_init]],
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    target_landmarks = torch.tensor(
        target_landmarks_np,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    landmark_weights = (
        create_landmark_weights(
            DEVICE
        )
    )

    print("\n=== LANDMARK WEIGHTS ===")
    print("Eyes:  5.0x")
    print("Nose:  2.0x")
    print("Mouth: 2.0x")
    print("Other: 1.0x")

    print("\n=== INITIAL CAMERA ===")

    print(
        "Scale:",
        scale_init,
    )

    print(
        "Translation:",
        tx_init,
        ty_init,
    )

    # ========================================================
    # Stage 1
    # ========================================================

    # Fit time = stage 1 + stage 2 optimisation only.
    fit_start = time.perf_counter()

    print(
        "\n=== STAGE 1: CAMERA + POSE ==="
    )

    optimizer = torch.optim.Adam(
        [
            rotations,
            scale,
            camera_translation,
        ],
        lr=LR_STAGE_1,
    )

    for step in range(
        STAGE_1_STEPS
    ):

        optimizer.zero_grad()

        vertices = gnm(
            identity,
            expression,
            rotations,
            model_translation,
        )

        landmarks_3d = (
            extract_correspondence(
                vertices,
                indices,
                weights,
            )
        )

        projected = project_landmarks(
            landmarks_3d,
            scale,
            camera_translation,
        )

        landmark_loss = (
            calculate_landmark_loss(
                projected,
                target_landmarks,
                landmark_weights,
            )
        )

        rotation_regularization = (
            0.001
            * torch.mean(
                rotations ** 2
            )
        )

        loss = (
            landmark_loss
            + rotation_regularization
        )

        loss.backward()

        optimizer.step()

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f} | "
                f"landmark {landmark_loss.item():.4f}"
            )

    # ========================================================
    # Stage 2
    # ========================================================

    print(
        "\n=== STAGE 2: IDENTITY + EXPRESSION ==="
    )

    optimizer = torch.optim.Adam(
        [
            identity,
            expression,
            rotations,
            scale,
            camera_translation,
        ],
        lr=LR_STAGE_2,
    )

    for step in range(
        STAGE_2_STEPS
    ):

        optimizer.zero_grad()

        vertices = gnm(
            identity,
            expression,
            rotations,
            model_translation,
        )

        landmarks_3d = (
            extract_correspondence(
                vertices,
                indices,
                weights,
            )
        )

        projected = project_landmarks(
            landmarks_3d,
            scale,
            camera_translation,
        )

        landmark_loss = (
            calculate_landmark_loss(
                projected,
                target_landmarks,
                landmark_weights,
            )
        )

        identity_regularization = (
            0.0001
            * torch.mean(
                identity ** 2
            )
        )

        expression_regularization = (
            0.0001
            * torch.mean(
                expression ** 2
            )
        )

        rotation_regularization = (
            0.001
            * torch.mean(
                rotations ** 2
            )
        )

        loss = (
            landmark_loss
            + identity_regularization
            + expression_regularization
            + rotation_regularization
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            identity.clamp_(
                -3.0,
                3.0,
            )

            expression.clamp_(
                -3.0,
                3.0,
            )

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f} | "
                f"landmark {landmark_loss.item():.4f}"
            )

    fit_seconds = time.perf_counter() - fit_start

    print(f"\nFit time (stage 1 + 2): {fit_seconds:.1f} s")

    # ========================================================
    # Final GNM mesh
    # ========================================================

    with torch.no_grad():

        final_vertices = gnm(
            identity,
            expression,
            rotations,
            model_translation,
        )

        final_landmarks_3d = (
            extract_correspondence(
                final_vertices,
                indices,
                weights,
            )
        )

        projected = project_landmarks(
            final_landmarks_3d,
            scale,
            camera_translation,
        )

    # Remove batch dimension.

    vertices_np = (
        final_vertices[0]
        .detach()
        .cpu()
        .numpy()
    )

    triangles_np = (
        gnm.triangles
        .detach()
        .cpu()
        .numpy()
    )

    projected_np = (
        projected[0]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # Error analysis
    # ========================================================

    errors = print_error_statistics(
        target_landmarks_np,
        projected_np,
    )

    # ========================================================
    # Parameter diagnostics
    # ========================================================

    print("\n=== PARAMETER RANGE ===")

    identity_np = (
        identity
        .detach()
        .cpu()
        .numpy()
    )

    expression_np = (
        expression
        .detach()
        .cpu()
        .numpy()
    )

    rotations_np = (
        rotations
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "Identity min/max:",
        identity_np.min(),
        identity_np.max(),
    )

    print(
        "Expression min/max:",
        expression_np.min(),
        expression_np.max(),
    )

    print(
        "Rotation min/max:",
        rotations_np.min(),
        rotations_np.max(),
    )

    identity_clamped = np.sum(
        np.isclose(
            np.abs(identity_np),
            3.0,
            atol=1e-4,
        )
    )

    expression_clamped = np.sum(
        np.isclose(
            np.abs(expression_np),
            3.0,
            atol=1e-4,
        )
    )

    print(
        "Identity parameters at ±3:",
        identity_clamped,
        "/",
        identity_np.size,
    )

    print(
        "Expression parameters at ±3:",
        expression_clamped,
        "/",
        expression_np.size,
    )

    # ========================================================
    # Optional metrics file (for batch runs)
    # ========================================================

    if args.metrics_json:

        metrics = {
            "image": str(image_path),
            "landmarks_file": str(LANDMARK_FILE),
            "image_width": width,
            "image_height": height,
            "fit_seconds": fit_seconds,
            "identity_at_limit": int(identity_clamped),
            "identity_total": int(identity_np.size),
            "expression_at_limit": int(expression_clamped),
            "expression_total": int(expression_np.size),
            "errors_px": errors.tolist(),
            "target_landmarks_px": target_landmarks_np.tolist(),
            "regions": {
                name: list(map(int, region_indices))
                for name, region_indices in REGIONS.items()
            },
        }

        Path(args.metrics_json).write_text(
            json.dumps(metrics)
        )

        print("Metrics:", args.metrics_json)

    # ========================================================
    # 2D landmark diagnostic
    # ========================================================

    diagnostic = (
        create_diagnostic_image(
            image,
            target_landmarks_np,
            projected_np,
            errors,
        )
    )

    diagnostic_path = (
        out_dir
        / "gnm_fit_diagnostic.png"
    )

    cv2.imwrite(
        str(diagnostic_path),
        diagnostic,
    )

    # ========================================================
    # 2D GNM mesh overlay
    # ========================================================

    print("\n=== MESH OVERLAY ===")

    camera_translation_np = (
        camera_translation[0]
        .detach()
        .cpu()
        .numpy()
    )

    mesh_overlay = (
        create_mesh_overlay(
            image,
            vertices_np,
            triangles_np,
            scale.item(),
            camera_translation_np,
        )
    )

    mesh_overlay_path = (
        out_dir
        / "gnm_fit_mesh_overlay.png"
    )

    cv2.imwrite(
        str(mesh_overlay_path),
        mesh_overlay,
    )

    print(
        "Mesh overlay:",
        mesh_overlay_path,
    )

    # ========================================================
    # Export OBJ
    # ========================================================

    print("\n=== OBJ EXPORT ===")

    obj_path = (
        out_dir
        / "gnm_fit.obj"
    )

    mesh = export_gnm_mesh(
        vertices_np,
        triangles_np,
        obj_path,
    )

    print(
        "OBJ:",
        obj_path,
    )

    print(
        "Mesh vertices:",
        len(mesh.vertices),
    )

    print(
        "Mesh faces:",
        len(mesh.faces),
    )

    # ========================================================
    # Save parameters
    # ========================================================

    params_path = (
        out_dir
        / "gnm_fit.npz"
    )

    np.savez(
        params_path,

        identity=(
            identity
            .detach()
            .cpu()
            .numpy()
        ),

        expression=(
            expression
            .detach()
            .cpu()
            .numpy()
        ),

        rotations=(
            rotations
            .detach()
            .cpu()
            .numpy()
        ),

        translation=(
            model_translation
            .detach()
            .cpu()
            .numpy()
        ),

        camera_scale=(
            scale
            .detach()
            .cpu()
            .numpy()
        ),

        camera_translation=(
            camera_translation
            .detach()
            .cpu()
            .numpy()
        ),
    )

    print(
        "Parameters:",
        params_path,
    )

    # ========================================================
    # Interactive viewer
    # ========================================================

    if args.view:

        print(
            "\n=== INTERACTIVE VIEWER ==="
        )

        print(
            "Close the viewer window to continue."
        )

        mesh.show()

    # ========================================================
    # Final
    # ========================================================

    print("\n=== FINAL OUTPUTS ===")

    print(
        "Diagnostic:",
        diagnostic_path,
    )

    print(
        "Mesh overlay:",
        mesh_overlay_path,
    )

    print(
        "OBJ:",
        obj_path,
    )

    print(
        "Parameters:",
        params_path,
    )


if __name__ == "__main__":
    main()
