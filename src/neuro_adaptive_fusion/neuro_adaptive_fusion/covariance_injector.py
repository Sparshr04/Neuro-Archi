"""covariance_injector.py — ROS 2 node: Mock MAVLink Covariance Injection.

Subscribes to the adapted covariance diagonal from ``dnn_inference`` and
logs the values to the console, simulating the MAVLink parameter injection
to the Cube Orange+ flight controller.

In production, this node would call MAVSDK to set EKF2 parameters:
    EKF2_GPS_P_NOISE, EKF2_GPS_V_NOISE, EKF2_BARO_NOISE, etc.

Topic flow
----------
  /neuro/covariance_correction  →  [CovarianceInjector]  →  console (mock MAVLink)
"""

from __future__ import annotations

from typing import Final

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


N_CHANNELS: Final[int] = 6

# Channel labels for readable logging
CHANNEL_LABELS: Final[tuple[str, ...]] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
)


class CovarianceInjector(Node):
    """Mock node that logs the adapted covariance (simulates MAVLink TX)."""

    def __init__(self) -> None:
        super().__init__("covariance_injector")

        self._sub = self.create_subscription(
            Float32MultiArray,
            "/neuro/covariance_correction",
            self._correction_cb,
            qos_profile=10,
        )

        self._msg_count: int = 0
        self.get_logger().info("CovarianceInjector ready  [mock — console logging only]")

    def _correction_cb(self, msg: Float32MultiArray) -> None:
        """Log the incoming R_adapt diagonal, simulating FCU injection."""
        r_adapt = np.array(msg.data, dtype=np.float32)
        self._msg_count += 1

        # Format: readable channel-by-channel breakdown
        parts = [f"{lbl}={val:.6f}" for lbl, val in zip(CHANNEL_LABELS, r_adapt)]
        detail = ", ".join(parts)

        self.get_logger().info(f"[#{self._msg_count}] Injecting Covariance Delta: [{detail}]")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CovarianceInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
