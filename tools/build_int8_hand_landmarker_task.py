#!/usr/bin/env python
"""Build an int8 Hand Landmarker .task bundle from the official float16 bundle.

This script:
1) Downloads (or reuses) the official float16 hand_landmarker.task bundle.
2) Extracts both internal TFLite models.
3) Quantizes each model to int8 activations using TensorFlow Lite calibrator.
4) Repackages them into models/hand_landmarker_int8.task.

Note:
- This uses TensorFlow internal calibrator APIs.
- Calibration data is synthetic by default (random samples), which may reduce accuracy.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.optimize.calibrator import Calibrator


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

FLOAT16_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

MODEL_ENTRIES = ("hand_detector.tflite", "hand_landmarks_detector.tflite")


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def download_if_missing(url: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing source task: {destination}")
        return
    print(f"Downloading source task from: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded: {destination}")


def read_task_entries(task_path: pathlib.Path) -> dict[str, bytes]:
    with zipfile.ZipFile(task_path, "r") as zf:
        names = set(zf.namelist())
        missing = [entry for entry in MODEL_ENTRIES if entry not in names]
        if missing:
            raise FileNotFoundError(
                f"Task bundle missing expected entries: {missing}. Found: {sorted(names)}"
            )
        return {entry: zf.read(entry) for entry in MODEL_ENTRIES}


def _safe_shape(raw_shape: np.ndarray) -> list[int]:
    shape = []
    for dim in raw_shape.tolist():
        if dim is None or dim <= 0:
            shape.append(1)
        else:
            shape.append(int(dim))
    return shape


def _sample_tensor(rng: np.random.Generator, shape: list[int], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape, dtype=np.float32).astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(info.min, -128)
        high = min(info.max, 127)
        # randint high is exclusive
        return rng.integers(low, high + 1, size=shape, dtype=dtype)

    raise TypeError(f"Unsupported representative dtype: {dtype}")


def quantize_model_to_int8(
    model_content: bytes,
    sample_count: int,
    seed: int,
    strict_integer_only: bool,
) -> bytes:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    input_details = interpreter.get_input_details()

    specs: list[tuple[list[int], np.dtype]] = []
    for detail in input_details:
        shape = _safe_shape(detail["shape"])
        dtype = np.dtype(detail["dtype"])
        specs.append((shape, dtype))

    rng = np.random.default_rng(seed)

    def representative_dataset():
        for _ in range(sample_count):
            sample = [_sample_tensor(rng, shape, dtype) for shape, dtype in specs]
            yield sample

    calibrator = Calibrator(model_content)
    try:
        quantized = calibrator.calibrate_and_quantize(
            representative_dataset,
            input_type=tf.float32,
            output_type=tf.float32,
            allow_float=not strict_integer_only,
            activations_type=tf.int8,
            bias_type=tf.int32,
            resize_input=True,
            disable_per_channel=False,
            disable_per_channel_quantization_for_dense_layers=False,
        )
    except Exception as exc:
        mode = "strict integer-only" if strict_integer_only else "int8 with float fallback"
        raise RuntimeError(f"Quantization failed in {mode} mode: {exc}") from exc

    return quantized


def summarize_tensor_dtypes(model_content: bytes) -> dict[str, int]:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    details = interpreter.get_tensor_details()
    counts: dict[str, int] = {}
    for detail in details:
        name = np.dtype(detail["dtype"]).name
        counts[name] = counts.get(name, 0) + 1
    return counts


def write_task_bundle(entries: dict[str, bytes], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-task",
        type=pathlib.Path,
        default=root / "models" / "hand_landmarker_float16.task",
        help="Path to source float16 task bundle.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=FLOAT16_TASK_URL,
        help="Download URL used when --source-task is missing.",
    )
    parser.add_argument(
        "--output-task",
        type=pathlib.Path,
        default=root / "models" / "hand_landmarker_int8.task",
        help="Path to write the int8 task bundle.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Representative samples per model for calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic representative data.",
    )
    parser.add_argument(
        "--allow-float-fallback",
        action="store_true",
        help="Allow fallback float ops if strict integer quantization fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_task = args.source_task.resolve()
    output_task = args.output_task.resolve()
    strict_integer_only = not args.allow_float_fallback

    print("=== Build int8 Hand Landmarker task ===")
    print(f"Source task : {source_task}")
    print(f"Output task : {output_task}")
    print(f"Samples     : {args.samples}")
    print(f"Strict int8 : {strict_integer_only}")

    download_if_missing(args.source_url, source_task)

    source_entries = read_task_entries(source_task)
    quantized_entries: dict[str, bytes] = {}

    for i, name in enumerate(MODEL_ENTRIES):
        print(f"\nQuantizing {name}...")
        quantized = quantize_model_to_int8(
            source_entries[name],
            sample_count=args.samples,
            seed=args.seed + i,
            strict_integer_only=strict_integer_only,
        )
        dtype_counts = summarize_tensor_dtypes(quantized)
        print(f"Tensor dtypes: {dtype_counts}")
        quantized_entries[name] = quantized

    write_task_bundle(quantized_entries, output_task)
    print(f"\nWrote int8 task bundle: {output_task}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
