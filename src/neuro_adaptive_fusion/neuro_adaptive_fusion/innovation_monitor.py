"""innovation_monitor.py — ROS 2 node: EKF Innovation Monitor with Feature Extraction.

Subscribes to the EKF innovation stream from the Cube Orange+ and extracts
the 4-feature vector (Eq 27–29) for the Supervisor DNN.

Pipeline
--------
  /ekf/innovations  →  [deque W=64]  →  decimation(8)  →  feature extraction  →  /neuro/features

Features (per channel, m=6):
  1. Sample Mean     μ̂_W
  2. Sample Variance σ̂²_W
  3. Normalised Innovation Squared (NIS)
  4. Zero-Crossing Rate (ZCR)

Output: Float32MultiArray of shape (24,) = 4 features × 6 channels.
"""

from __future__ import annotations

import collections
import threading
from typing import Final

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE: Final[int] = 64  # W — sliding window length
N_CHANNELS: Final[int] = 6  # m — 6-DOF IMU innovation channels
DECIMATION: Final[int] = 8  # publish features every N-th sample
N_FEATURES_PER_CH: Final[int] = 4  # mean, var, NIS, ZCR

# Baseline measurement-noise variance R₀ (diagonal), must match generate_data.py
R0_VAR: Final[np.ndarray] = np.array(
    [
        0.012**2,  # accel x  (m/s²)²
        0.012**2,  # accel y
        0.012**2,  # accel z
        0.004**2,  # gyro  x  (rad/s)²
        0.004**2,  # gyro  y
        0.004**2,  # gyro  z
    ],
    dtype=np.float32,
)


class InnovationMonitor(Node):
    """Collects EKF innovations into a sliding window and publishes features.

    Thread-safety: all mutable state (``_buffer``, ``_msg_count``) is
    guarded by ``_lock``.  ROS 2 single-threaded executor serialises
    callbacks by default, but the lock ensures correctness if a
    multi-threaded executor is ever used.
    """

    def __init__(self) -> None:
        super().__init__("innovation_monitor")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("window_size", WINDOW_SIZE)
        self.declare_parameter("n_channels", N_CHANNELS)
        self.declare_parameter("decimation", DECIMATION)

        self._win: int = self.get_parameter("window_size").value
        self._nch: int = self.get_parameter("n_channels").value
        self._dec: int = self.get_parameter("decimation").value

        # ── State (guarded by _lock) ─────────────────────────────────────────
        self._lock = threading.Lock()
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self._win)
        self._msg_count: int = 0

        # Pre-compute R₀⁻¹ for NIS
        self._R0_inv: np.ndarray = (1.0 / R0_VAR).astype(np.float32)

        # ── Pub / Sub ────────────────────────────────────────────────────────
        self._sub = self.create_subscription(
            Float32MultiArray,
            "/ekf/innovations",
            self._innovation_cb,
            qos_profile=10,
        )
        self._pub = self.create_publisher(
            Float32MultiArray,
            "/neuro/features",
            qos_profile=10,
        )

        self.get_logger().info(
            f"InnovationMonitor ready  "
            f"[W={self._win}, channels={self._nch}, decimation={self._dec}]"
        )

    # ── Callback ─────────────────────────────────────────────────────────────

    def _innovation_cb(self, msg: Float32MultiArray) -> None:
        """Buffer each innovation sample; extract features every ``_dec`` messages."""
        sample = np.array(msg.data, dtype=np.float32)

        if sample.shape[0] != self._nch:
            self.get_logger().warn(
                f"Expected {self._nch} channels, got {sample.shape[0]} — skipping",
                throttle_duration_sec=5.0,
            )
            return

        with self._lock:
            self._buffer.append(sample)
            self._msg_count += 1

            # Only extract when buffer is full AND on the decimation cycle
            if len(self._buffer) < self._win:
                return
            if self._msg_count % self._dec != 0:
                return

            # Snapshot under lock → release lock before compute
            window = np.array(self._buffer, dtype=np.float32)  # (W, C)

        features = self._extract_features(window)
        self._publish_features(features)

    # ── Feature Extraction (Eq 27–29) ────────────────────────────────────────

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """Compute the 4-feature vector ξ_j from the innovation window.

        Parameters
        ----------
        window : (W, C) float32 array

        Returns
        -------
        features : (C * 4,) = (24,) float32 array
            Concatenation: [μ̂_0..5, σ̂²_0..5, NIS_0..5, ZCR_0..5]
        """
        # 1. Sample Mean
        mu: np.ndarray = np.mean(window, axis=0)  # (C,)

        # 2. Sample Variance (unbiased)
        var: np.ndarray = np.var(window, axis=0, ddof=1)  # (C,)

        # 3. Normalised Innovation Squared:  NIS_ch = (1/W) Σ (ỹ²_ch / R₀_ch)
        nis: np.ndarray = np.mean(window**2, axis=0) * self._R0_inv  # (C,)

        # 4. Zero-Crossing Rate
        signs: np.ndarray = np.sign(window)  # (W, C)
        crossings: np.ndarray = np.abs(np.diff(signs, axis=0))  # (W-1, C)
        zcr: np.ndarray = np.mean(crossings, axis=0) / 2.0  # (C,) in [0, 1]

        return np.concatenate([mu, var, nis, zcr]).astype(np.float32)

    # ── Publisher ────────────────────────────────────────────────────────────

    def _publish_features(self, features: np.ndarray) -> None:
        """Publish the feature vector as Float32MultiArray."""
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label="features",
                size=len(features),
                stride=len(features),
            ),
        ]
        msg.data = features.tolist()
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = InnovationMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
