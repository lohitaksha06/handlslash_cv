# Hill Climb Gesture-Based Control

Simple OpenCV-based Hill Climb Racing game control using MediaPipe Hand Landmarker.

## Quantized TFLite Pipeline

This project now uses the MediaPipe Tasks `HandLandmarker` pipeline backed by a
TFLite model bundle (`hand_landmarker.task`).

- Default mode: float16.
- Int8 path is temporarily disabled in code while conversion is stabilized.

### Model Selection

Use environment variables:

- `HAND_LANDMARKER_VARIANT=float16` (default: `float16`)
- `HAND_LANDMARKER_MODEL_PATH=...` to use any local `.task` model file
- `HAND_LANDMARKER_MODEL_URL=...` optional download URL when model file is missing

Notes:

- Public MediaPipe hand landmarker bundle is currently available as float16.
- Int8 references are intentionally disabled for now.

## Run

Install dependencies:

```bash
pip install mediapipe opencv-python
```

Start the controller:

```bash
python main.py
```

Run (auto-download float16 from MediaPipe):

```bash
set HAND_LANDMARKER_VARIANT=float16
python main.py
```

## Startup Self-Check Output

At startup, the app prints model verification details, including:

- requested variant
- model source
- resolved model path
- model size
- model last-modified timestamp (UTC)
- SHA-256 hash

This helps confirm that the project is actually running the expected model.

Press `q` to quit.
