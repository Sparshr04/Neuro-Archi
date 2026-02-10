"""dnn_inference.py — ROS 2 node: TFLite Supervisor DNN Inference.

Loads the INT8-quantised ``neuro_adapter.tflite`` model and runs inference
on each incoming feature vector to produce the adaptive covariance
correction ΔR.

Equation 30 (Springer chapter):
    R_adapt = R₀ + α · ΔR_dnn

Pipeline
--------
  /neuro/features  →  [DNNInference]  →  /neuro/covariance_correction

Output: Float32MultiArray of shape (6,) — the full adapted diagonal R_adapt.
"""

from __future__ import annotations

import pathlib
import threading
import time
from typing import Final

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

# ── TFLite runtime import (RPi vs host fallback) ────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────
N_CHANNELS: Final[int] = 6
N_FEATURES: Final[int] = 24  # 4 features × 6 channels
DEFAULT_MODEL_PATH: Final[str] = "ml_core/models/neuro_adapter.tflite"
DEFAULT_ALPHA: Final[float] = 1.0  # blending gain α (Eq 30)

# Baseline measurement-noise variance R₀ (diagonal)
R0_VAR: Final[np.ndarray] = np.array(
    [
        0.012**2,  # accel x
        0.012**2,  # accel y
        0.012**2,  # accel z
        0.004**2,  # gyro  x
        0.004**2,  # gyro  y
        0.004**2,  # gyro  z
    ],
    dtype=np.float32,
)


class DNNInference(Node):
    """Run the quantised Supervisor DNN and publish R_adapt.

    Thread-safety: the TFLite interpreter is NOT thread-safe.  All access
    is serialised by ``_lock``.  Under the default SingleThreadedExecutor
    this is redundant but guarantees correctness under MultiThreadedExecutor.
    """

    def __init__(self) -> None:
        super().__init__("dnn_inference")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("alpha", DEFAULT_ALPHA)

        model_path = pathlib.Path(
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self._alpha: float = self.get_parameter("alpha").get_parameter_value().double_value

        # ── Load TFLite interpreter (once) ───────────────────────────────────
        if not model_path.exists():
            self.get_logger().fatal(f"Model not found: {model_path}")
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self._interpreter = TFLiteInterpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_detail = self._interpreter.get_output_details()[0]

        # Quantisation parameters for INT8 input
        self._input_dtype: np.dtype = self._input_detail["dtype"]
        qp = self._input_detail.get("quantization_parameters", {})
        self._input_scale: float = float(qp.get("scales", [1.0])[0])
        self._input_zp: int = int(qp.get("zero_points", [0])[0])

        self._lock = threading.Lock()

        self.get_logger().info(
            f"DNNInference ready  "
            f"[model={model_path}, α={self._alpha}, "
            f"input_dtype={self._input_dtype.__name__}]"
        )

        # ── Pub / Sub ────────────────────────────────────────────────────────
        self._sub = self.create_subscription(
            Float32MultiArray,
            "/neuro/features",
            self._feature_cb,
            qos_profile=10,
        )
        self._pub = self.create_publisher(
            Float32MultiArray,
            "/neuro/covariance_correction",
            qos_profile=10,
        )

    # ── Callback ─────────────────────────────────────────────────────────────

    def _feature_cb(self, msg: Float32MultiArray) -> None:
        """Run inference on incoming feature vector and publish R_adapt."""
        features = np.array(msg.data, dtype=np.float32)

        if features.shape[0] != N_FEATURES:
            self.get_logger().warn(
                f"Expected {N_FEATURES} features, got {features.shape[0]} — skipping",
                throttle_duration_sec=5.0,
            )
            return

        # ── Quantise input if model expects INT8 ────────────────────────────
        input_tensor = features.reshape(1, N_FEATURES)
        if self._input_dtype == np.int8:
            input_tensor = np.clip(
                np.round(input_tensor / self._input_scale) + self._input_zp,
                -128,
                127,
            ).astype(np.int8)
        elif self._input_dtype == np.uint8:
            input_tensor = np.clip(
                np.round(input_tensor / self._input_scale) + self._input_zp,
                0,
                255,
            ).astype(np.uint8)

        # ── Inference (serialised) ──────────────────────────────────────────
        with self._lock:
            t0: int = time.monotonic_ns()
            self._interpreter.set_tensor(self._input_detail["index"], input_tensor)
            self._interpreter.invoke()
            delta_r = (
                self._interpreter.get_tensor(self._output_detail["index"])
                .flatten()
                .astype(np.float32)
            )
            dt_us: float = (time.monotonic_ns() - t0) / 1e3

        # ── Eq 30:  R_adapt = R₀ + α · ΔR_dnn ──────────────────────────────
        delta_r_clamped: np.ndarray = np.clip(delta_r, 0.0, None)  # ΔR ≥ 0
        r_adapt: np.ndarray = R0_VAR + self._alpha * delta_r_clamped

        # ── Publish ─────────────────────────────────────────────────────────
        out_msg = Float32MultiArray()
        out_msg.layout.dim = [
            MultiArrayDimension(
                label="r_adapt_diag",
                size=N_CHANNELS,
                stride=N_CHANNELS,
            ),
        ]
        out_msg.data = r_adapt.tolist()
        self._pub.publish(out_msg)

        self.get_logger().debug(
            f"Inference: {dt_us:.0f} µs | ΔR={delta_r_clamped} | R_adapt={r_adapt}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DNNInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
