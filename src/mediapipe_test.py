import argparse
import sys

from preflight import FACE_MODEL, PROJECT_ROOT, require

require(face_model=True, packages=("cv2", "mediapipe"))

import cv2
import mediapipe as mp


parser = argparse.ArgumentParser(
    description="Draw MediaPipe face landmarks on an image."
)
parser.add_argument(
    "--image",
    default=str(PROJECT_ROOT / "data" / "tests" / "test.jpg"),
    help="Path to input face image (default: data/tests/test.jpg)",
)
args = parser.parse_args()

IMAGE_PATH = args.image
MODEL_PATH = str(FACE_MODEL)


# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    sys.exit(f"Could not load image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)


with FaceLandmarker.create_from_options(options) as landmarker:

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = landmarker.detect(mp_image)


if not result.face_landmarks:
    raise RuntimeError("No face detected.")


landmarks = result.face_landmarks[0]

print("Number of landmarks:", len(landmarks))

for i, landmark in enumerate(landmarks[:10]):
    print(
        i,
        "x:", landmark.x,
        "y:", landmark.y,
        "z:", landmark.z,
    )
# Draw landmarks
for i, landmark in enumerate(landmarks):
    x = int(landmark.x * image.shape[1])
    y = int(landmark.y * image.shape[0])

    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)

    if i < 468:
        cv2.putText(
            image,
            str(i),
            (x + 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.,
            (255, 255, 255),
            1,
        )

cv2.imshow("MediaPipe Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
