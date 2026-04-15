import os
# Reduce TensorFlow C++ backend log noise.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable oneDNN path for more consistent behavior across machines.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl and other logging
import logging
import absl.logging
# Remove the default absl logging hook so startup output stays clean.
logging.root.removeHandler(absl.logging._absl_handler)
# Avoid pre-init warning spam from absl.
absl.logging._warn_preinit_stderr = False

# OpenCV handles webcam capture and on-screen drawing.
import cv2
# MediaPipe provides hand landmark detection/tracking.
import mediapipe as mp
# time is used for startup delay and timestamp generation.
import time
# hashlib is used to print model SHA-256 for integrity/provenance checks.
import hashlib
# Path is used for robust model path handling.
from pathlib import Path
# urlretrieve downloads a model if it is missing locally.
from urllib.request import urlretrieve
# Import pyautogui for mouse control
import pyautogui
# Use deque for storing trail points
from collections import deque
import numpy as np

# Set pyautogui to not fail out if move to corner, and disable delay
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Screen size for mapping
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

from collections import deque
import numpy as np

# Trail for the ninja slice effect
slash_trail = deque(maxlen=20)  # Max length of the tail

# Import arrow-key scan code constants used by the game.
from directkeys import right_pressed,left_pressed
# Import helper functions that synthesize key down/up events on Windows.
from directkeys import PressKey, ReleaseKey


# Map game actions to keyboard scan codes.
break_key_pressed=left_pressed
accelerato_key_pressed=right_pressed

# Absolute path to this script's directory.
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
# Tracks keys currently held down by this script.
current_key_pressed = set()
is_mouse_down = False
previous_gesture = None  # Track previous gesture to avoid repeated key presses

# Trail data for the fruit ninja blade effect
# stores tuples of (col, row) up to maxlen points
slash_trail = deque(maxlen=15)

frame_skip = 0  # Process every nth frame for better performance
# Last timestamp passed to VIDEO-mode inference.
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
    # Ensure the destination directory exists before file checks/download.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # No-op when the model file is already present.
    if model_path.exists():
        return
    # If no downloadable source exists, fail with a clear error.
    if model_url in ("", "custom"):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    # Download official model artifact.
    print(f"Downloading TFLite task model to {model_path}...")
    urlretrieve(model_url, model_path)


def sha256_file(file_path):
    """Compute SHA-256 hash for model provenance/integrity checks."""
    # Incremental hashing avoids loading the full model into memory.
    digest = hashlib.sha256()
    # Read bytes from disk because hash functions operate on bytes.
    with open(file_path, "rb") as model_file:
        while True:
            # 1 MiB chunk size balances speed and memory use.
            chunk = model_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    # Return the digest as a hex string.
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
    # Capture frame dimensions to convert normalized landmarks -> pixel space.
    h, w, _ = image.shape
    # Keep pixel points so we can draw skeleton edges by landmark index.
    points = []
    for lm in hand_landmarks:
        # Convert normalized (0..1) coordinate to pixel coordinate.
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        # Draw each landmark as a small filled circle.
        cv2.circle(image, (px, py), 2, (0, 255, 255), cv2.FILLED)

    # Draw line segments that form the hand skeleton.
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], (0, 200, 0), 1)

# Open the default system webcam.
video=cv2.VideoCapture(0)

if not video.isOpened():
    # Abort early if camera is unavailable.
    raise RuntimeError("Could not open webcam. Check camera permissions/device availability.")

# Optimize camera settings for better performance
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 60)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

MODEL_PATH, MODEL_URL = resolve_model_path_and_url()
ensure_model_exists(MODEL_PATH, MODEL_URL)
print_model_self_check(MODEL_PATH, HAND_LANDMARKER_VARIANT, MODEL_URL)

# HandLandmarker task setup:
# - VIDEO mode enables tracking optimizations between frames.
# - num_hands=1 keeps latency lower for this game controller scenario.
options = HandLandmarkerOptions(
    # Path to selected .task model bundle.
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    # VIDEO mode uses temporal tracking and requires timestamps.
    running_mode=VisionRunningMode.VIDEO,
    # Single-hand setup for low-latency controller behavior.
    num_hands=1,
    # Confidence thresholds for detection/presence/tracking.
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

try:
    # Context manager handles native resource lifecycle.
    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            # Skip frames for better performance
            frame_skip += 1
            # Read one frame from webcam.
            ret, image = video.read()
            if not ret:
                # Frame grab can fail transiently; continue loop.
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
                    # Draw landmarks on display frame for visual feedback.
                    draw_hand_landmarks(image, hand_landmarks)

                    # Quantize normalized landmarks into an 8-bit grid.
                    for id, lm in enumerate(hand_landmarks):
                        qx, qy = quantize_landmark_coords(lm)
                        lmList.append([id, qx, qy])

                    import math
                    # Extract index tip for cursor position
                    index_tip = hand_landmarks[8]
                    
                    # Flip x-axis since camera is mirrored
                    cursor_x = int((1.0 - index_tip.x) * SCREEN_WIDTH) 
                    cursor_y = int(index_tip.y * SCREEN_HEIGHT)

                    # Clamp to screen bounds
                    cursor_x = max(0, min(SCREEN_WIDTH - 1, cursor_x))
                    cursor_y = max(0, min(SCREEN_HEIGHT - 1, cursor_y))

                    # Move the mouse cursor instantly
                    pyautogui.moveTo(cursor_x, cursor_y)

                    # Check which fingers are open
                    fingers = []
                    if len(lmList) != 0:
                        # Thumb (using x-axis relationship)
                        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                        # Remaining checking y-axis for tip vs joint 
                        for id in range(1, 5):
                            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                                
                    total_open_fingers = fingers.count(1)
                    current_gesture = "NONE"
                    
                    # Store tracking point for the trail
                    frame_w = image.shape[1]
                    frame_h = image.shape[0]
                    trail_px = int(index_tip.x * frame_w)
                    trail_py = int(index_tip.y * frame_h)

                    # If mostly closed (0 to 1 open fingers), consider it a fist / cup (for selecting/grabbing)
                    if total_open_fingers <= 1:
                        current_gesture = "SELECT (FIST)"
                        slash_trail.append((trail_px, trail_py))
                        if not is_mouse_down:
                            pyautogui.mouseDown()
                            is_mouse_down = True
                    else:
                        current_gesture = "HOVER"
                        slash_trail.clear()  # Clear trail when not grabbing
                        if is_mouse_down:
                            pyautogui.mouseUp()
                            is_mouse_down = False
                    previous_gesture = current_gesture

            # Draw the ninja slash trail
            if len(slash_trail) > 1:
                # Convert deque to a list for easier iteration
                pts = list(slash_trail)
                for i in range(1, len(pts)):
                    # Calculate thickness that fades from the start to the end
                    progress = i / len(pts)
                    thickness = int(12 * progress) + 2
                    glow_thickness = thickness + 8
                    
                    # Yellow/Orange glow (BGR format)
                    cv2.line(image, pts[i - 1], pts[i], (0, 165, 255), glow_thickness, cv2.LINE_AA)
                    # White core
                    cv2.line(image, pts[i - 1], pts[i], (255, 255, 255), thickness, cv2.LINE_AA)

            # Update gesture display outside the processing loop to prevent flickering
            if previous_gesture == "SELECT (FIST)":
                cv2.rectangle(image, (20, 300), (350, 425), (0, 0, 255), cv2.FILLED)  # Grab/select box
                cv2.putText(image, "SELECT!", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            elif previous_gesture == "HOVER":
                cv2.rectangle(image, (20, 300), (350, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, "HOVER", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

            # Show the annotated frame in an OpenCV window.
            cv2.imshow("Frame", image)
            # Exit loop when q is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    # Release any remaining pressed keys before closing.
    for key in current_key_pressed:
        ReleaseKey(key)

    if is_mouse_down:
        pyautogui.mouseUp()
        
    video.release()
    cv2.destroyAllWindows()

