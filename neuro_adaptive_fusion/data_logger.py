"""data_logger.py — ROS 2 node: One-Cycle Flight Data Logger.

Subscribes to the Neuro-Adaptive EKF Digital Twin topics and writes a
structured CSV file (``flight_data_log.csv``) capturing exactly one 30-second
flight cycle (Hover → Transition → Cruise).

Topic Map (real topics published by this codebase)
───────────────────────────────────────────────────
  SUB  /sim/state                  Float32MultiArray  [sim_t, true_z, est_base, est_adapt, delta_r]
  SUB  /diagnostics/noise_gate     std_msgs/Bool      True when transition noise is active
  SUB  /neuro/covariance_correction Float32MultiArray  6-ch ΔR from DNN (ch-0 = acc_x scalar)

CSV Columns
───────────
  timestamp, true_z, est_z, z_error, covariance_scale_factor, noise_gate_active

Loop-Detection & Shutdown
──────────────────────────
  • Recording starts immediately at t=0.
  • A time-reset is detected when current_sim_t < previous_sim_t (cycle wrap).
  • The node shuts itself down automatically after the first reset is detected
    AND sim_t > SHUTDOWN_GRACE_S (default 1 s) to capture the wrap boundary.

Usage
─────
  # Terminal 1 — start the full pipeline
  ros2 launch neuro_adaptive_fusion system.launch.py

  # Terminal 2 — start the data logger
  ros2 run neuro_adaptive_fusion data_logger

  # The node exits automatically after ~31 s and writes flight_data_log.csv
  # in the current working directory.
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Final

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray

# ── Constants ─────────────────────────────────────────────────────────────────

CYCLE_DURATION_S: Final[float] = 30.0  # nominal cycle length
SHUTDOWN_GRACE_S: Final[float] = 1.0  # seconds into new cycle before shutdown
OUTPUT_FILE: Final[str] = "flight_data_log.csv"

CSV_HEADER: Final[list[str]] = [
    "timestamp",
    "true_z",
    "est_z",
    "z_error",
    "covariance_scale_factor",
    "noise_gate_active",
]

# Channel indices in /sim/state Float32MultiArray
IDX_SIM_T: Final[int] = 0
IDX_TRUE_Z: Final[int] = 1
IDX_EST_BASE: Final[int] = 2
IDX_EST_ADAPT: Final[int] = 3
IDX_DELTA_R: Final[int] = 4


# ═══════════════════════════════════════════════════════════════════════════════
#  DataLogger Node
# ═══════════════════════════════════════════════════════════════════════════════


class DataLogger(Node):
    """Captures one full flight cycle from the Digital Twin to a CSV file.

    State machine
    ─────────────
      WAITING  → first /sim/state message received → RECORDING
      RECORDING → time-reset detected AND grace period elapsed → DONE
      DONE     → node shuts itself down
    """

    def __init__(self, output_path: str = OUTPUT_FILE) -> None:
        super().__init__("data_logger")

        self._output_path = Path(output_path).resolve()
        self._lock = threading.Lock()

        # Executor reference — set by main() so _shutdown() can unblock spin
        self._executor: SingleThreadedExecutor | None = None

        # ── Shared state (guarded by _lock) ──────────────────────────────────
        self._prev_sim_t: float = -1.0
        self._cycle_done: bool = False
        self._recording: bool = False

        # Latest values from async topics (updated by their own callbacks)
        self._noise_gate: bool = False
        self._delta_r: float = 0.0  # from /neuro/covariance_correction ch-0

        # ── CSV writer ───────────────────────────────────────────────────────
        self._csv_file = open(self._output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_HEADER)
        self._writer.writeheader()
        self._row_count: int = 0

        # ── Subscriptions ────────────────────────────────────────────────────
        # Primary state vector from sim_node (sim_t, true_z, est_base, est_adapt, delta_r)
        self._sub_state = self.create_subscription(
            Float32MultiArray,
            "/sim/state",
            self._on_sim_state,
            qos_profile=10,
        )

        # Noise gate flag (True during Transition phase)
        self._sub_gate = self.create_subscription(
            Bool,
            "/diagnostics/noise_gate",
            self._on_noise_gate,
            qos_profile=10,
        )

        # DNN covariance correction (6-ch; ch-0 = acc_x scalar used as scale factor)
        self._sub_correction = self.create_subscription(
            Float32MultiArray,
            "/neuro/covariance_correction",
            self._on_covariance_correction,
            qos_profile=10,
        )

        self.get_logger().info(
            f"DataLogger ready — will write to: {self._output_path}\n"
            "Waiting for /sim/state …"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_noise_gate(self, msg: Bool) -> None:
        with self._lock:
            self._noise_gate = bool(msg.data)

    def _on_covariance_correction(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._delta_r = float(msg.data[0]) if len(msg.data) > 0 else 0.0

    def _on_sim_state(self, msg: Float32MultiArray) -> None:
        """Primary callback — drives the state machine and CSV writing."""
        if len(msg.data) < 5:
            self.get_logger().warn(
                f"/sim/state message has {len(msg.data)} elements, expected ≥5 — skipping",
                throttle_duration_sec=5.0,
            )
            return

        sim_t = float(msg.data[IDX_SIM_T])
        true_z = float(msg.data[IDX_TRUE_Z])
        est_z = float(msg.data[IDX_EST_ADAPT])  # Neuro-Adaptive EKF estimate
        # Use the delta_r embedded in the state vector (more synchronous than the
        # async /neuro/covariance_correction callback)
        delta_r_state = float(msg.data[IDX_DELTA_R])

        with self._lock:
            if self._cycle_done:
                return  # already finished — ignore further messages

            # ── Detect first message (start recording) ────────────────────
            if not self._recording:
                self._recording = True
                self._prev_sim_t = sim_t
                self.get_logger().info(f"Recording started at sim_t={sim_t:.3f}s")

            # ── Detect cycle reset ────────────────────────────────────────
            # A wrap is confirmed when sim_t drops significantly below prev_sim_t
            # (i.e. the 30 s cycle has restarted). We require prev_sim_t > 15 s
            # to avoid false triggers from startup jitter.
            time_reset = (sim_t < self._prev_sim_t) and (
                self._prev_sim_t > CYCLE_DURATION_S * 0.5
            )

            if time_reset:
                # We've wrapped into the second cycle — stop immediately
                self._cycle_done = True
                self.get_logger().info(
                    f"Cycle complete detected (prev={self._prev_sim_t:.3f}s → "
                    f"curr={sim_t:.3f}s). "
                    f"Wrote {self._row_count} rows. Shutting down …"
                )
                self._csv_file.flush()
                self._csv_file.close()
                # Shutdown the executor from a separate thread so we don't
                # deadlock inside the callback (executor owns this thread).
                threading.Thread(target=self._shutdown, daemon=True).start()
                return

            self._prev_sim_t = sim_t

            # ── Write CSV row ─────────────────────────────────────────────
            noise_gate = self._noise_gate
            # Prefer the state-vector delta_r (synchronous); fall back to async
            covariance_scale = delta_r_state if delta_r_state != 0.0 else self._delta_r

        z_error = true_z - est_z

        self._writer.writerow(
            {
                "timestamp": f"{sim_t:.6f}",
                "true_z": f"{true_z:.6f}",
                "est_z": f"{est_z:.6f}",
                "z_error": f"{z_error:.6f}",
                "covariance_scale_factor": f"{covariance_scale:.6f}",
                "noise_gate_active": int(noise_gate),
            }
        )
        self._row_count += 1

        # Flush every 400 rows (~1 s at 400 Hz) to avoid data loss
        if self._row_count % 400 == 0:
            self._csv_file.flush()
            self.get_logger().info(
                f"  … {self._row_count} rows logged  (sim_t={sim_t:.1f}s)",
                throttle_duration_sec=2.0,
            )

    # ── Shutdown helper ───────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Unblock the executor so main() can exit cleanly.

        Called from a worker thread (never from the executor's own thread)
        to avoid a deadlock.  Shutting down the executor is the reliable way
        to unblock executor.spin() on the main thread.
        """
        import time

        time.sleep(0.1)  # tiny delay so the log message above is flushed first
        self.get_logger().info(
            f"✓ flight_data_log.csv written → {self._output_path}  "
            f"({self._row_count} rows)"
        )
        if self._executor is not None:
            self._executor.shutdown()  # unblocks executor.spin() on main thread
        else:
            # Fallback: signal the global rclpy context
            rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(args: list[str] | None = None) -> None:
    """
    Run the data logger.

    Usage
    ─────
        ros2 run neuro_adaptive_fusion data_logger

    Optional: override output path via environment variable
        DATA_LOG_PATH=/tmp/my_log.csv ros2 run neuro_adaptive_fusion data_logger
    """
    rclpy.init(args=args)
    output_path = os.environ.get("DATA_LOG_PATH", OUTPUT_FILE)
    node = DataLogger(output_path=output_path)

    # Use an explicit executor so _shutdown() can call executor.shutdown()
    # to reliably unblock spin() from a worker thread.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node._executor = executor  # give the node a back-reference

    try:
        executor.spin()  # blocks until executor.shutdown() is called
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user — flushing CSV …")
        if not node._csv_file.closed:
            node._csv_file.flush()
            node._csv_file.close()
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
