"""quantize.py — Full INT8 quantization for Raspberry Pi 5 deployment.

Converts the trained Supervisor DNN (SavedModel) to a fully-quantized
INT8 TFLite flatbuffer suitable for inference via ``tflite-runtime``
on the RPi 5 companion computer.

Quantization strategy (per paper requirements):
    • Full Integer Quantization — both weights AND activations are INT8
    • Representative dataset drawn from actual training features for
      accurate activation range calibration
    • Input tensor:  INT8  (quantised on-device, no float pre-processing)
    • Output tensor: FLOAT32  (ΔR scale factors need full precision)

Usage
-----
    python -m ml_core.quantize
    python -m ml_core.quantize --model ml_core/models/supervisor_dnn \\
                               --data-dir ml_core/data \\
                               --output ml_core/models/neuro_adapter.tflite
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import tensorflow as tf


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "ml_core/models/supervisor_dnn"
DEFAULT_DATA_DIR = "ml_core/data"
DEFAULT_OUTPUT = "ml_core/models/neuro_adapter.tflite"
N_CALIBRATION_SAMPLES: int = 300


# ═══════════════════════════════════════════════════════════════════════════════
# Representative Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════════


def make_representative_dataset(
    data_dir: pathlib.Path,
    n_samples: int = N_CALIBRATION_SAMPLES,
):
    """Yield representative input samples for INT8 activation calibration.

    The TFLite converter calls this generator during quantization to
    determine the dynamic range of each activation tensor.  We sample
    uniformly from the training features so the calibration covers
    hover, transition, AND cruise regimes.

    Yields
    ------
    list[np.ndarray]
        Single-element list containing one input sample with shape (1, 24).
    """
    features_path = data_dir / "features.npy"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {features_path}\n"
            "Run `python -m ml_core.generate_data` first."
        )

    X = np.load(features_path).astype(np.float32)
    n_total = len(X)
    n_samples = min(n_samples, n_total)

    rng = np.random.default_rng(0)
    indices = rng.choice(n_total, size=n_samples, replace=False)

    for idx in indices:
        sample = X[idx : idx + 1]  # (1, 24)
        yield [sample]


# ═══════════════════════════════════════════════════════════════════════════════
# Quantization
# ═══════════════════════════════════════════════════════════════════════════════


def quantize(
    model_path: pathlib.Path = pathlib.Path(DEFAULT_MODEL_PATH),
    data_dir: pathlib.Path = pathlib.Path(DEFAULT_DATA_DIR),
    output_path: pathlib.Path = pathlib.Path(DEFAULT_OUTPUT),
    n_calibration: int = N_CALIBRATION_SAMPLES,
) -> None:
    """Convert SavedModel → fully-quantized INT8 TFLite flatbuffer.

    Steps
    -----
    1. Load the SavedModel exported by train_model.py
    2. Configure the TFLite converter for full integer quantization
    3. Provide a representative dataset for activation range calibration
    4. Set input type to INT8, output type to FLOAT32
    5. Serialize and write the .tflite file
    """
    print(f"[quantize] Loading model from {model_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))

    # ── Full Integer Quantization ────────────────────────────────────────────
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for activation calibration
    converter.representative_dataset = lambda: make_representative_dataset(data_dir, n_calibration)

    # Restrict ops to INT8-only builtins (ensures full quantization)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Input: INT8  (quantised on-device by the inference node)
    # Output: FLOAT32 (ΔR values need full precision for covariance injection)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    print(f"[quantize] Converting with {n_calibration} calibration samples …")
    tflite_model = converter.convert()

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_kb = output_path.stat().st_size / 1024
    print(f"\n{'═' * 55}")
    print(f"  INT8 TFLite model → {output_path}")
    print(f"  Size:  {size_kb:.1f} KB")
    print(f"{'═' * 55}")

    # ── Verify: load and inspect ─────────────────────────────────────────────
    _verify_tflite(output_path)


def _verify_tflite(tflite_path: pathlib.Path) -> None:
    """Load the quantized model and print tensor details for verification."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"\n  Input  → shape={inp['shape']}, dtype={inp['dtype'].__name__}")
    if "quantization_parameters" in inp:
        qp = inp["quantization_parameters"]
        if len(qp.get("scales", [])) > 0:
            print(f"           scale={qp['scales'][0]:.6f}, zero_point={qp['zero_points'][0]}")

    print(f"  Output → shape={out['shape']}, dtype={out['dtype'].__name__}")
    if "quantization_parameters" in out:
        qp = out["quantization_parameters"]
        if len(qp.get("scales", [])) > 0:
            print(f"           scale={qp['scales'][0]:.6f}, zero_point={qp['zero_points'][0]}")

    # Quick inference test with zeros
    test_input = np.zeros(inp["shape"], dtype=inp["dtype"])
    interpreter.set_tensor(inp["index"], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(out["index"])
    print(f"\n  Smoke test (zeros input) → output: {test_output.flatten()}")
    print(f"  ✓ Model loads and runs successfully")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize Supervisor DNN to INT8 TFLite for RPi 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to SavedModel directory."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory with features.npy for calibration.",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Output .tflite file path."
    )
    parser.add_argument(
        "--n-calibration",
        type=int,
        default=N_CALIBRATION_SAMPLES,
        help="Number of representative samples for calibration.",
    )
    args = parser.parse_args()

    quantize(
        model_path=pathlib.Path(args.model),
        data_dir=pathlib.Path(args.data_dir),
        output_path=pathlib.Path(args.output),
        n_calibration=args.n_calibration,
    )
