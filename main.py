import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl and other logging
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

import cv2
import mediapipe as mp
import time
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from directkeys import right_pressed,left_pressed
from directkeys import PressKey, ReleaseKey


break_key_pressed=left_pressed
accelerato_key_pressed=right_pressed

PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------
# MODEL CONFIGURATION
# ------------------------------
# 1) HAND_LANDMARKER_VARIANT:
#    - "float16" -> uses official MediaPipe float16 model bundle
#
# NOTE: int8 path is temporarily disabled while we stabilize conversion.
#
# 2) HAND_LANDMARKER_MODEL_PATH:
#    Optional custom absolute or relative path to a .task model file.
#
# 3) HAND_LANDMARKER_MODEL_URL:
#    Optional URL used for download when requested model file is not present.
HAND_LANDMARKER_VARIANT = os.getenv("HAND_LANDMARKER_VARIANT", "float16").strip().lower()
HAND_LANDMARKER_MODEL_PATH = os.getenv("HAND_LANDMARKER_MODEL_PATH", "").strip()
HAND_LANDMARKER_MODEL_URL = os.getenv("HAND_LANDMARKER_MODEL_URL", "").strip()

FLOAT16_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FLOAT16_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker_float16.task"

# ------------------------------
# PERFORMANCE SETTINGS
# ------------------------------
# INFERENCE_SIZE:
#   Frame is resized before inference to reduce per-frame compute cost.
# FRAME_PROCESS_STRIDE:
#   Process every Nth frame (e.g., 2 means process every second frame).
# LANDMARK_QUANT_LEVELS:
#   8-bit quantization grid for landmark coordinates (0..255).
INFERENCE_SIZE = (320, 240)
FRAME_PROCESS_STRIDE = 2
LANDMARK_QUANT_LEVELS = 255  # 8-bit coordinate quantization

# Small startup delay helps users switch to the game window before input starts.
time.sleep(2.0)

# Runtime state used by the gesture controller loop.
current_key_pressed = set()
previous_gesture = None  # Track previous gesture to avoid repeated key presses
frame_skip = 0  # Process every nth frame for better performance
last_timestamp_ms = 0

# Aliases to keep task API access concise and readable in workshop demos.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Connection pairs used for drawing hand skeleton lines.
try:
    HAND_CONNECTIONS = tuple(mp.solutions.hands.HAND_CONNECTIONS)
except AttributeError:
    # Fallback for builds where `mediapipe.solutions` is not exposed.
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    )

# Landmark index map for fingertips: thumb, index, middle, ring, pinky.
tipIds=[4,8,12,16,20]


def resolve_model_path_and_url():
    """Resolve which model file to load and where it comes from.

    Returns:
        Tuple[pathlib.Path, str]: (model_path, model_source)
            - model_source is either:
              - "custom" (already local file)
              - an HTTP URL (download required if file missing)
    """
    if HAND_LANDMARKER_MODEL_PATH:
        custom_path = Path(HAND_LANDMARKER_MODEL_PATH)
        if not custom_path.is_absolute():
            custom_path = PROJECT_ROOT / custom_path
        custom_path = custom_path.resolve()
        if not custom_path.exists():
            raise FileNotFoundError(f"Configured model path does not exist: {custom_path}")
        return custom_path, "custom"

    if HAND_LANDMARKER_VARIANT == "int8":
        print("Int8 is temporarily disabled; falling back to float16 model.")

    if HAND_LANDMARKER_VARIANT in ("float16", "int8"):
        return FLOAT16_MODEL_PATH, FLOAT16_MODEL_URL

    raise ValueError("HAND_LANDMARKER_VARIANT must be 'float16' for now.")


def ensure_model_exists(model_path, model_url):
    """Ensure the requested model exists locally; download only when allowed."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return
    if model_url in ("", "custom"):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    print(f"Downloading TFLite task model to {model_path}...")
    urlretrieve(model_url, model_path)


def sha256_file(file_path):
    """Compute SHA-256 hash for model provenance/integrity checks."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as model_file:
        while True:
            chunk = model_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def print_model_self_check(model_path, variant, model_source):
    """Print a clear startup report so workshop attendees can verify model identity."""
    stat_info = model_path.stat()
    model_hash = sha256_file(model_path)
    print("\n=== Hand Landmarker Model Self-Check ===")
    print(f"Variant requested   : {variant}")
    print(f"Model source       : {model_source}")
    print(f"Model path         : {model_path}")
    print(f"Model size (bytes) : {stat_info.st_size}")
    print(f"Model modified UTC : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(stat_info.st_mtime))}")
    print(f"Model SHA-256      : {model_hash}")
    print("========================================\n")


def get_timestamp_ms():
    """Generate strictly increasing timestamps for VIDEO mode inference calls.

    MediaPipe VIDEO mode expects monotonically increasing frame timestamps.
    """
    global last_timestamp_ms
    ts = int(time.perf_counter() * 1000)
    if ts <= last_timestamp_ms:
        ts = last_timestamp_ms + 1
    last_timestamp_ms = ts
    return ts


def quantize_landmark_coords(landmark):
    """Map normalized float coordinates into an 8-bit integer grid.

    Example:
        x = 0.5 -> int(0.5 * 255) = 127

    This keeps gesture logic lightweight and deterministic.
    """
    # Clamp values in case the model emits tiny out-of-range numbers.
    x = min(max(landmark.x,     0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return (
        int(x * LANDMARK_QUANT_LEVELS),
        int(y * LANDMARK_QUANT_LEVELS),
    )


def draw_hand_landmarks(image, hand_landmarks):
    """Render landmark points and hand connection lines on the display frame."""
    h, w, _ = image.shape
    points = []
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(image, (px, py), 2, (0, 255, 255), cv2.FILLED)

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], (0, 200, 0), 1)

video=cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions/device availability.")

# Optimize camera settings for better performance
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

MODEL_PATH, MODEL_URL = resolve_model_path_and_url()
ensure_model_exists(MODEL_PATH, MODEL_URL)
print_model_self_check(MODEL_PATH, HAND_LANDMARKER_VARIANT, MODEL_URL)

# HandLandmarker task setup:
# - VIDEO mode enables tracking optimizations between frames.
# - num_hands=1 keeps latency lower for this game controller scenario.
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

try:
    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            # Skip frames for better performance
            frame_skip += 1
            ret, image = video.read()
            if not ret:
                continue

            if frame_skip % FRAME_PROCESS_STRIDE == 0:
                # 1) Preprocess frame for inference: resize and convert BGR -> RGB.
                resized_frame = cv2.resize(image, INFERENCE_SIZE)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # 2) Wrap numpy frame in MediaPipe Image and run VIDEO-mode inference.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = hand_landmarker.detect_for_video(mp_image, get_timestamp_ms())

                # 3) Build compact landmark list for gesture logic.
                lmList = []
                if results.hand_landmarks:
                    # We only use the first detected hand because num_hands=1.
                    hand_landmarks = results.hand_landmarks[0]
                    draw_hand_landmarks(image, hand_landmarks)

                    # Quantize normalized landmarks into an 8-bit grid.
                    for id, lm in enumerate(hand_landmarks):
                        qx, qy = quantize_landmark_coords(lm)
                        lmList.append([id, qx, qy])

                # 4) Convert landmarks to finger-open/finger-closed states.
                fingers = []
                if len(lmList) != 0:
                    # Thumb rule uses x-axis relationship for this camera orientation.
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Remaining fingers use y-axis tip-vs-joint relationship.
                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    total = fingers.count(1)

                    # 5) Map finger-count to game gesture.
                    current_gesture = "NONE"
                    if total == 0:
                        current_gesture = "BRAKE"
                    elif total == 5:
                        current_gesture = "GAS"

                    # Only update keys if gesture has changed
                    if current_gesture != previous_gesture:
                        if current_gesture == "BRAKE":
                            for key in current_key_pressed:
                                ReleaseKey(key)
                            current_key_pressed.clear()
                            PressKey(break_key_pressed)
                            current_key_pressed.add(break_key_pressed)
                        elif current_gesture == "GAS":
                            for key in current_key_pressed:
                                ReleaseKey(key)
                            current_key_pressed.clear()
                            PressKey(accelerato_key_pressed)
                            current_key_pressed.add(accelerato_key_pressed)
                        else:  # "NONE"
                            for key in current_key_pressed:
                                ReleaseKey(key)
                            current_key_pressed.clear()
                        previous_gesture = current_gesture

            # Update gesture display outside the processing loop to prevent flickering
            if previous_gesture == "BRAKE":
                cv2.rectangle(image, (20, 300), (270, 425), (0, 0, 255), cv2.FILLED)  # Red box for brake
                cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            elif previous_gesture == "GAS":
                cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, " GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

            cv2.imshow("Frame", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    # Release any remaining pressed keys before closing.
    for key in current_key_pressed:
        ReleaseKey(key)

    video.release()
    cv2.destroyAllWindows()

