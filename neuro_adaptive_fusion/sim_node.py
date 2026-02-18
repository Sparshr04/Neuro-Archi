# """sim_node.py — SITL Digital Twin for Neuro-Adaptive EKF Validation.

# Generates 3-phase noise profiles (Hover → Transition Corridor → Cruise) at
# 400 Hz and runs two parallel 1-D Kalman estimators — Baseline (fixed R) vs
# Neuro-Adaptive (dynamic R from DNN) — with a real-time matplotlib dashboard.

# Threading model
# ───────────────
#   Main thread   → matplotlib event loop  (macOS requires GUI on main thread)
#   Daemon thread → rclpy.spin             (publishes innovations, receives ΔR)

# ROS 2 Topics
# ────────────
#   PUB  /ekf/innovations             Float32MultiArray  (6 ch, 400 Hz)
#   SUB  /neuro/covariance_correction  Float32MultiArray  (6 ch, from DNN)

# Noise Phases (30 s cycle, looping)
# ──────────────────────────────────
#   Phase A  [0–10 s]   Hover       σ² = 0.1
#   Phase B  [10–20 s]  Transition  σ² = 2.0  +  2.0·sin(2π·25·t)  vibration
#   Phase C  [20–30 s]  Cruise      σ² = 0.1

# Usage
# ─────
#     ros2 run neuro_adaptive_fusion sim_node
#     ros2 launch neuro_adaptive_fusion system.launch.py
# """


"""sim_node.py — SITL Digital Twin for Neuro-Adaptive EKF Validation.

Updates:
- LINEAR RESPONSE: Replaced quadratic penalty with linear to prevent "explosive" corrections.
- SMOOTHING: Added a Low-Pass Filter (LPF) to delta_r so the correction is smooth, not jagged.
- TUNED GAINS: Adjusted threshold and gain for a balanced response.
"""

from __future__ import annotations

import collections
import math
import threading
from typing import Final

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

SIM_RATE_HZ: Final[int] = 400
DT: Final[float] = 1.0 / SIM_RATE_HZ
N_CH: Final[int] = 6
PLOT_HISTORY: Final[int] = 2400
PLOT_REFRESH_MS: Final[int] = 50

# ── Phase boundaries ────────────────────────────────────────────────────────
T_HOVER_END: Final[float] = 10.0
T_TRANSITION_END: Final[float] = 20.0
T_CYCLE: Final[float] = 30.0

# ── Noise parameters ────────────────────────────────────────────────────────
SIGMA2_NOMINAL: Final[float] = 0.1
SIGMA2_TRANSITION: Final[float] = 2.0
VIB_FREQ_HZ: Final[float] = 25.0
VIB_AMP: Final[float] = 2.0

# ── Kalman parameters ───────────────────────────────────────────────────────
R_BASELINE: Final[float] = 0.1
Q_PROCESS: Final[float] = 0.001

# ── Mock Neural Network Parameters ──────────────────────────────────────────
# 1. Gate: Innovation below this is ignored (Silence)
NN_GATE_THRESHOLD: Final[float] = 1.5

# 2. Gain: Linear response (Correction = Gain * Excess_Noise)
#    Reduced from "Quadratic" to prevent explosions.
NN_DAMPING_GAIN: Final[float] = 0.5

# 3. Smoothing: Low-Pass Filter factor (0.0 = frozen, 1.0 = no smoothing)
#    0.1 means "take 10% new value, keep 90% old value". Makes it smooth.
NN_SMOOTHING_ALPHA: Final[float] = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
#  1-D Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════


class KalmanFilter1D:
    """Minimal scalar Kalman filter."""

    __slots__ = ("x", "p", "q")

    def __init__(self, x0: float = 0.0, p0: float = 1.0, q: float = Q_PROCESS):
        self.x: float = x0
        self.p: float = p0
        self.q: float = q

    def predict(self) -> None:
        self.p += self.q

    def update(self, z: float, r: float) -> float:
        self.predict()
        k = self.p / (self.p + r)
        self.x += k * (z - self.x)
        self.p *= 1.0 - k
        return self.x

    def reset(self) -> None:
        self.x = 0.0
        self.p = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation Node
# ═══════════════════════════════════════════════════════════════════════════════


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_node")

        self._lock = threading.Lock()
        self._sim_t: float = 0.0
        self._rng = np.random.default_rng(seed=42)
        self._running: bool = True

        # Estimators
        self._kf_base = KalmanFilter1D()
        self._kf_adapt = KalmanFilter1D()

        # State for smoothing
        self._dr_smooth: float = 0.0

        # History buffers
        self._h_time = collections.deque(maxlen=PLOT_HISTORY)
        self._h_truth = collections.deque(maxlen=PLOT_HISTORY)
        self._h_err_base = collections.deque(maxlen=PLOT_HISTORY)
        self._h_err_adapt = collections.deque(maxlen=PLOT_HISTORY)
        self._h_innov = collections.deque(maxlen=PLOT_HISTORY)
        self._h_delta_r = collections.deque(maxlen=PLOT_HISTORY)

        self._pub = self.create_publisher(Float32MultiArray, "/ekf/innovations", 10)
        self._timer = self.create_timer(DT, self._tick)

        self.get_logger().info(f"Digital Twin Active: Smoothness={NN_SMOOTHING_ALPHA}")

    # ── MOCK NN LOGIC (Smoothed) ─────────────────────────────────────────────

    def _emulate_nn_prediction(self, innovation: float) -> float:
        """
        Simulates NN with Linear Response + Smoothing.
        """
        mag = abs(innovation)

        # 1. Calculate Target Correction (Instantaneous)
        if mag < NN_GATE_THRESHOLD:
            target_dr = 0.0
        else:
            excess = mag - NN_GATE_THRESHOLD
            # Linear response: Proportional to error
            target_dr = NN_DAMPING_GAIN * excess

        # Clamp max to prevent graph destruction
        target_dr = min(target_dr, 3.0)

        # 2. Apply Smoothing (Low Pass Filter)
        # new_val = alpha * target + (1 - alpha) * old_val
        self._dr_smooth = (NN_SMOOTHING_ALPHA * target_dr) + (
            (1.0 - NN_SMOOTHING_ALPHA) * self._dr_smooth
        )

        return self._dr_smooth

    # ── PHYSICS TICK ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        t = self._sim_t

        # Cycle Reset
        if t >= T_CYCLE:
            self._sim_t = 0.0
            self._kf_base.reset()
            self._kf_adapt.reset()
            self._dr_smooth = 0.0  # Reset filter state
            self.get_logger().info("─── cycle reset ───")
            return

        # 1. Ground Truth & Noise
        x_true = 0.0
        if t < T_HOVER_END:
            noise = float(self._rng.normal(0.0, math.sqrt(SIGMA2_NOMINAL)))
        elif t < T_TRANSITION_END:
            base = float(self._rng.normal(0.0, math.sqrt(SIGMA2_TRANSITION)))
            vib = VIB_AMP * math.sin(2.0 * math.pi * VIB_FREQ_HZ * t)
            noise = base + vib
        else:
            noise = float(self._rng.normal(0.0, math.sqrt(SIGMA2_NOMINAL)))

        measurement = x_true + noise

        # 2. Baseline EKF
        est_base = self._kf_base.update(measurement, R_BASELINE)

        # 3. Compute Innovation
        innovation = measurement - est_base

        # 4. Mock NN Inference (Smoothed)
        current_dr = self._emulate_nn_prediction(innovation)

        # 5. Adaptive EKF
        r_adapt = R_BASELINE + current_dr
        est_adapt = self._kf_adapt.update(measurement, r_adapt)

        # Publish
        msg = Float32MultiArray()
        msg.data = [float(innovation)] * N_CH
        self._pub.publish(msg)

        # Record
        with self._lock:
            self._h_time.append(t)
            self._h_truth.append(x_true)
            self._h_err_base.append(est_base - x_true)
            self._h_err_adapt.append(est_adapt - x_true)
            self._h_innov.append(innovation)
            self._h_delta_r.append(current_dr)

        self._sim_t += DT

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "t": list(self._h_time),
                "truth": list(self._h_truth),
                "err_base": list(self._h_err_base),
                "err_adapt": list(self._h_err_adapt),
                "innov": list(self._h_innov),
                "delta_r": list(self._h_delta_r),
            }

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Dashboard
# ═══════════════════════════════════════════════════════════════════════════════


def _run_dashboard(node: SimNode) -> None:
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Neuro-Adaptive EKF · SITL Digital Twin", fontsize=15, color="#00e5ff")
    fig.subplots_adjust(hspace=0.32, top=0.93, bottom=0.06)

    def _decorate(ax):
        ax.axvspan(T_HOVER_END, T_TRANSITION_END, alpha=0.08, color="#ff9800")
        ax.grid(True, alpha=0.15)

    # Graph 1: Error
    ax1.set_title("Position Estimation Error (m)", fontsize=11, color="#b0bec5")
    (ln_base,) = ax1.plot([], [], color="#f44336", lw=0.8, alpha=0.85, label="Baseline")
    (ln_adapt,) = ax1.plot([], [], color="#00bcd4", lw=1.3, label="Neuro-Adaptive")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-6, 6)
    _decorate(ax1)

    # Graph 2: Innovation
    ax2.set_title(
        "Innovation Sequence (Raw Sensor Noise)", fontsize=11, color="#b0bec5"
    )
    (ln_innov,) = ax2.plot([], [], color="#ffc107", lw=0.5, alpha=0.9)
    ax2.set_ylim(-5, 5)
    _decorate(ax2)

    # Graph 3: Correction
    ax3.set_title(
        "Neural Network Covariance Correction (ΔR)", fontsize=11, color="#b0bec5"
    )
    (ln_dr,) = ax3.plot([], [], color="#ce93d8", lw=1.5)  # Thicker line for visibility
    ax3.axhline(
        NN_GATE_THRESHOLD, color="white", ls=":", alpha=0.3, label="Gate Trigger"
    )
    ax3.set_ylim(-0.2, 4.0)  # Adjusted scale to fit the smoothed response
    ax3.set_ylabel("Added Variance (m²)")
    _decorate(ax3)

    def _refresh(_):
        d = node.snapshot()
        t = d["t"]
        if len(t) < 2:
            return

        ln_base.set_data(t, d["err_base"])
        ln_adapt.set_data(t, d["err_adapt"])
        ln_innov.set_data(t, d["innov"])
        ln_dr.set_data(t, d["delta_r"])

        ax1.set_xlim(max(0, t[-1] - PLOT_HISTORY * DT), t[-1] + 0.5)
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=PLOT_REFRESH_MS)
    timer.add_callback(_refresh, 0)
    timer.start()
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    try:
        _run_dashboard(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
