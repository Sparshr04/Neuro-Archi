from __future__ import annotations

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

"""dnn_inference.py — ROS 2 node: TFLite Supervisor DNN Inference."""


import pathlib
import threading
import time
from typing import Final

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ament_index_python.packages import get_package_share_directory
from tflite_runtime.interpreter import Interpreter


# ── Constants ────────────────────────────────────────────────────────────────
N_CHANNELS: Final[int] = 6
N_FEATURES: Final[int] = 24
DEFAULT_MODEL_PATH: Final[str] = ""  # ← IMPORTANT CHANGE
DEFAULT_ALPHA: Final[float] = 1.0

R0_VAR: Final[np.ndarray] = np.array(
    [
        0.012**2,
        0.012**2,
        0.012**2,
        0.004**2,
        0.004**2,
        0.004**2,
    ],
    dtype=np.float32,
)


class DNNInference(Node):
    def __init__(self) -> None:
        super().__init__("dnn_inference")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("alpha", DEFAULT_ALPHA)

        self._alpha: float = (
            self.get_parameter("alpha").get_parameter_value().double_value
        )

        model_path_param = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )

        # ── Resolve model path (ROS-native) ──────────────────────────────────
        if model_path_param:
            model_path = pathlib.Path(model_path_param)
        else:
            pkg_share = pathlib.Path(
                get_package_share_directory("neuro_adaptive_fusion")
            )
            model_path = pkg_share / "models" / "neuro_adapter.tflite"

        # ── Load TFLite Interpreter ──────────────────────────────────────────
        if not model_path.exists():
            self.get_logger().fatal(f"Model not found: {model_path}")
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self.get_logger().info(f"✅ TFLite model loaded: {model_path}")

        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_detail = self._interpreter.get_output_details()[0]

        self._input_dtype = self._input_detail["dtype"]
        qp = self._input_detail.get("quantization_parameters", {})
        self._input_scale = float(qp.get("scales", [1.0])[0])
        self._input_zp = int(qp.get("zero_points", [0])[0])

        self._lock = threading.Lock()

        self.get_logger().info(
            f"DNNInference ready [α={self._alpha}, input_dtype={self._input_dtype}]"
        )

        # ── Pub / Sub ────────────────────────────────────────────────────────
        self._sub = self.create_subscription(
            Float32MultiArray,
            "/neuro/features",
            self._feature_cb,
            10,
        )

        self._pub = self.create_publisher(
            Float32MultiArray,
            "/neuro/covariance_correction",
            10,
        )

    # ── Callback ─────────────────────────────────────────────────────────────

    def _feature_cb(self, msg: Float32MultiArray) -> None:
        features = np.array(msg.data, dtype=np.float32)

        if features.shape[0] != N_FEATURES:
            self.get_logger().warn(
                f"Expected {N_FEATURES} features, got {features.shape[0]}",
                throttle_duration_sec=5.0,
            )
            return

        input_tensor = features.reshape(1, N_FEATURES)

        if self._input_dtype == np.int8:
            input_tensor = np.round(input_tensor / self._input_scale) + self._input_zp
            input_tensor = np.clip(input_tensor, -128, 127).astype(np.int8)
        elif self._input_dtype == np.uint8:
            input_tensor = np.round(input_tensor / self._input_scale) + self._input_zp
            input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)

        with self._lock:
            t0 = time.monotonic_ns()
            self._interpreter.set_tensor(self._input_detail["index"], input_tensor)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_detail["index"])
            delta_r = output.flatten().astype(np.float32)
            dt_us = (time.monotonic_ns() - t0) / 1e3

        r_adapt = R0_VAR + self._alpha * np.clip(delta_r, 0.0, None)

        msg_out = Float32MultiArray()
        msg_out.layout.dim = [
            MultiArrayDimension(
                label="r_adapt_diag",
                size=N_CHANNELS,
                stride=N_CHANNELS,
            )
        ]
        msg_out.data = r_adapt.tolist()
        self._pub.publish(msg_out)

        self.get_logger().debug(f"Inference {dt_us:.0f} µs | R={r_adapt}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DNNInference()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
