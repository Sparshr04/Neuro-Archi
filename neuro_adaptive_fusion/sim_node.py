"""sim_node.py — SITL Digital Twin for Neuro-Adaptive EKF Validation.

Generates 3-phase noise profiles (Hover → Transition Corridor → Cruise) at
400 Hz and runs two parallel 1-D Kalman estimators — Baseline (fixed R) vs
Neuro-Adaptive (dynamic R from DNN) — with a real-time matplotlib dashboard.

Threading model
───────────────
  Main thread   → matplotlib event loop  (macOS requires GUI on main thread)
  Daemon thread → rclpy.spin             (publishes innovations, receives ΔR)

ROS 2 Topics
────────────
  PUB  /ekf/innovations             Float32MultiArray  (6 ch, 400 Hz)
  SUB  /neuro/covariance_correction  Float32MultiArray  (6 ch, from DNN)

Noise Phases (30 s cycle, looping)
──────────────────────────────────
  Phase A  [0–10 s]   Hover       σ² = 0.1
  Phase B  [10–20 s]  Transition  σ² = 2.0  +  2.0·sin(2π·25·t)  vibration
  Phase C  [20–30 s]  Cruise      σ² = 0.1

Usage
─────
    ros2 run neuro_adaptive_fusion sim_node
    ros2 launch neuro_adaptive_fusion system.launch.py
"""

from __future__ import annotations

import collections
import math
import threading
from typing import Final

import matplotlib

matplotlib.use("TkAgg")  # must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import (  # noqa: E402
    Bool,
    Float32MultiArray,
    MultiArrayDimension,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

SIM_RATE_HZ: Final[int] = 400  # physics tick rate
DT: Final[float] = 1.0 / SIM_RATE_HZ  # time step (2.5 ms)
N_CH: Final[int] = 6  # 6-DOF innovation channels
PLOT_HISTORY: Final[int] = 2400  # samples visible ≈ 6 s
PLOT_REFRESH_MS: Final[int] = 50  # dashboard refresh interval

# ── Phase boundaries (seconds) ───────────────────────────────────────────────
T_HOVER_END: Final[float] = 10.0
T_TRANSITION_END: Final[float] = 20.0
T_CYCLE: Final[float] = 30.0

# ── Noise parameters ────────────────────────────────────────────────────────
SIGMA2_NOMINAL: Final[float] = 0.1  # hover / cruise variance
SIGMA2_TRANSITION: Final[float] = 2.0  # transition corridor variance
VIB_FREQ_HZ: Final[float] = 25.0  # vibration frequency
VIB_AMP: Final[float] = 2.0  # vibration amplitude

# ── Kalman filter parameters ────────────────────────────────────────────────
R_BASELINE: Final[float] = 0.1  # fixed measurement covariance
Q_PROCESS: Final[float] = 0.001  # process noise variance


# ═══════════════════════════════════════════════════════════════════════════════
#  1-D Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════


class KalmanFilter1D:
    """Minimal scalar Kalman filter.

    State model:   x_{k+1} = x_k  + w_k    w ~ N(0, Q)
    Obs   model:   z_k     = x_k  + v_k    v ~ N(0, R)
    """

    __slots__ = ("x", "p", "q")

    def __init__(
        self,
        x0: float = 0.0,
        p0: float = 1.0,
        q: float = Q_PROCESS,
    ) -> None:
        self.x: float = x0
        self.p: float = p0
        self.q: float = q

    def predict(self) -> None:
        """Propagate state (identity transition)."""
        self.p += self.q

    def update(self, z: float, r: float) -> float:
        """Incorporate measurement *z* with noise variance *r*.

        Returns the updated state estimate.
        """
        self.predict()
        k = self.p / (self.p + r)  # Kalman gain
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
    """SITL Digital Twin with twin estimators and real-time dashboard."""

    def __init__(self) -> None:
        super().__init__("sim_node")

        # ── mutable state (guarded by _lock) ─────────────────────────────────
        self._lock = threading.Lock()
        self._sim_t: float = 0.0
        self._rng = np.random.default_rng(seed=42)
        self._running: bool = True

        # Twin estimators
        self._kf_base = KalmanFilter1D()
        self._kf_adapt = KalmanFilter1D()

        # ΔR correction from DNN (written by subscriber callback)
        self._delta_r: float = 0.0

        # Rolling history for the dashboard
        self._h_time: collections.deque[float] = collections.deque(maxlen=PLOT_HISTORY)
        self._h_truth: collections.deque[float] = collections.deque(maxlen=PLOT_HISTORY)
        self._h_err_base: collections.deque[float] = collections.deque(
            maxlen=PLOT_HISTORY
        )
        self._h_err_adapt: collections.deque[float] = collections.deque(
            maxlen=PLOT_HISTORY
        )
        self._h_innov: collections.deque[float] = collections.deque(maxlen=PLOT_HISTORY)
        self._h_delta_r: collections.deque[float] = collections.deque(
            maxlen=PLOT_HISTORY
        )

        # ── ROS 2 interfaces ────────────────────────────────────────────────
        self._pub = self.create_publisher(
            Float32MultiArray,
            "/ekf/innovations",
            10,
        )
        # State vector for data_logger: [sim_t, true_z, est_base, est_adapt, delta_r]
        self._pub_state = self.create_publisher(
            Float32MultiArray,
            "/sim/state",
            10,
        )
        # Noise gate flag for data_logger (True during Transition phase)
        self._pub_gate = self.create_publisher(
            Bool,
            "/diagnostics/noise_gate",
            10,
        )
        self._sub = self.create_subscription(
            Float32MultiArray,
            "/neuro/covariance_correction",
            self._on_correction,
            10,
        )
        self._timer = self.create_timer(DT, self._tick)

        self.get_logger().info(
            f"SimNode SITL ready  [{SIM_RATE_HZ} Hz · "
            f"hover<{T_HOVER_END}s · transition<{T_TRANSITION_END}s · "
            f"cruise<{T_CYCLE}s · loop]"
        )

    # ── subscriber callback (from DNN inference) ─────────────────────────────

    def _on_correction(self, msg: Float32MultiArray) -> None:
        with self._lock:
            # Take channel 0 as the scalar correction for the 1-D demo
            self._delta_r = float(msg.data[0]) if len(msg.data) > 0 else 0.0

    # ── physics tick @ 400 Hz ────────────────────────────────────────────────

    def _tick(self) -> None:
        t = self._sim_t

        # ── cycle reset ──────────────────────────────────────────────────────
        if t >= T_CYCLE:
            self._sim_t = 0.0
            self._kf_base.reset()
            self._kf_adapt.reset()
            self.get_logger().info("─── cycle reset ───")
            return

        # ── ground truth ─────────────────────────────────────────────────────
        x_true: float = 0.0  # stable hover target position

        # ── noise generation (3 phases) ──────────────────────────────────────
        if t < T_HOVER_END:
            # Phase A: Hover — nominal Gaussian
            noise: float = float(self._rng.normal(0.0, math.sqrt(SIGMA2_NOMINAL)))

        elif t < T_TRANSITION_END:
            # Phase B: Transition Corridor — heavy noise + sinusoidal vibration
            #   noise = N(0, √2.0) + 2.0 · sin(2π · 25 · t)
            noise = float(
                self._rng.normal(0.0, math.sqrt(SIGMA2_TRANSITION))
            ) + VIB_AMP * math.sin(2.0 * math.pi * VIB_FREQ_HZ * t)

        else:
            # Phase C: Cruise — nominal Gaussian
            noise = float(self._rng.normal(0.0, math.sqrt(SIGMA2_NOMINAL)))

        measurement: float = x_true + noise

        # ── twin estimator updates ───────────────────────────────────────────
        # 1. Baseline: fixed R
        est_base: float = self._kf_base.update(measurement, R_BASELINE)

        # 2. Adaptive: R = R_base + ΔR  (ΔR from neural network)
        with self._lock:
            dr = max(self._delta_r, 0.0)  # ΔR ≥ 0
        r_adapt: float = R_BASELINE + dr
        est_adapt: float = self._kf_adapt.update(measurement, r_adapt)

        # ── publish innovations (all 6 channels = same scalar for demo) ──────
        innov_vec = np.full(N_CH, noise, dtype=np.float32)
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=N_CH, stride=N_CH),
        ]
        msg.data = innov_vec.tolist()
        self._pub.publish(msg)

        # ── publish /sim/state for data_logger ───────────────────────────────
        # Layout: [sim_t, true_z, est_base, est_adapt, delta_r]
        state_msg = Float32MultiArray()
        state_msg.layout.dim = [
            MultiArrayDimension(label="state", size=5, stride=5),
        ]
        state_msg.data = [
            float(t),
            float(x_true),
            float(est_base),
            float(est_adapt),
            float(dr),
        ]
        self._pub_state.publish(state_msg)

        # ── publish /diagnostics/noise_gate for data_logger ──────────────────
        gate_msg = Bool()
        gate_msg.data = bool(T_HOVER_END <= t < T_TRANSITION_END)
        self._pub_gate.publish(gate_msg)

        # ── record to history (under lock) ───────────────────────────────────
        with self._lock:
            self._h_time.append(t)
            self._h_truth.append(x_true)
            self._h_err_base.append(est_base - x_true)
            self._h_err_adapt.append(est_adapt - x_true)
            self._h_innov.append(noise)
            self._h_delta_r.append(dr)

        self._sim_t += DT

    # ── thread-safe snapshot for the dashboard ───────────────────────────────

    def snapshot(self) -> dict[str, list[float]]:
        """Return a copy of all time-series for plotting."""
        with self._lock:
            return {
                "t": list(self._h_time),
                "truth": list(self._h_truth),
                "err_base": list(self._h_err_base),
                "err_adapt": list(self._h_err_adapt),
                "innov": list(self._h_innov),
                "delta_r": list(self._h_delta_r),
            }

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Real-Time Dashboard (matplotlib — main thread)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_dashboard(node: SimNode) -> None:
    """3-subplot real-time dashboard.  Blocks on main thread."""
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
    )
    fig.suptitle(
        "Neuro-Adaptive EKF  ·  SITL Digital Twin",
        fontsize=15,
        fontweight="bold",
        color="#00e5ff",
    )
    fig.subplots_adjust(hspace=0.32, top=0.93, bottom=0.06, left=0.07, right=0.97)

    # helper — add phase shading + labels to every axes
    def _decorate(ax: plt.Axes) -> None:
        ax.axvspan(T_HOVER_END, T_TRANSITION_END, alpha=0.08, color="#ff9800")
        ax.grid(True, alpha=0.15)
        yl = ax.get_ylim()
        y_top = yl[1] * 0.88
        ax.text(
            T_HOVER_END / 2,
            y_top,
            "HOVER",
            ha="center",
            fontsize=8,
            color="#4caf50",
            alpha=0.5,
        )
        ax.text(
            (T_HOVER_END + T_TRANSITION_END) / 2,
            y_top,
            "TRANSITION",
            ha="center",
            fontsize=9,
            color="#ff9800",
            alpha=0.85,
            fontweight="bold",
        )
        ax.text(
            (T_TRANSITION_END + T_CYCLE) / 2,
            y_top,
            "CRUISE",
            ha="center",
            fontsize=8,
            color="#4caf50",
            alpha=0.5,
        )

    # ── Subplot 1: Position Estimation Error ─────────────────────────────────
    ax1.set_title("Position Estimation Error", fontsize=11, color="#b0bec5")
    ax1.set_ylabel("Error (m)")
    (ln_truth,) = ax1.plot([], [], color="#4caf50", lw=1.5, label="Ground Truth")
    (ln_base,) = ax1.plot(
        [], [], color="#f44336", lw=0.8, alpha=0.85, label="Baseline EKF"
    )
    (ln_adapt,) = ax1.plot([], [], color="#00bcd4", lw=1.3, label="Neuro-Adaptive EKF")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.5)
    ax1.set_ylim(-6, 6)
    ax1.axhline(0, color="#4caf50", ls="--", lw=0.5, alpha=0.3)
    _decorate(ax1)

    # ── Subplot 2: Innovation Sequence ───────────────────────────────────────
    ax2.set_title(
        "Innovation Sequence (Raw Sensor Noise)", fontsize=11, color="#b0bec5"
    )
    ax2.set_ylabel("Innovation")
    (ln_innov,) = ax2.plot([], [], color="#ffc107", lw=0.5, alpha=0.9)
    ax2.set_ylim(-8, 8)
    _decorate(ax2)

    # ── Subplot 3: Neural Network Correction ─────────────────────────────────
    ax3.set_title(
        "Neural Network Covariance Correction (ΔR)", fontsize=11, color="#b0bec5"
    )
    ax3.set_ylabel("ΔR")
    ax3.set_xlabel("Time (s)")
    (ln_dr,) = ax3.plot([], [], color="#ce93d8", lw=1.2)
    ax3.set_ylim(-0.05, 3.0)
    _decorate(ax3)

    # ── refresh callback ─────────────────────────────────────────────────────
    def _refresh(_frame: int) -> None:
        d = node.snapshot()
        t = d["t"]
        if len(t) < 2:
            return

        ln_truth.set_data(t, d["truth"])
        ln_base.set_data(t, d["err_base"])
        ln_adapt.set_data(t, d["err_adapt"])
        ln_innov.set_data(t, d["innov"])
        ln_dr.set_data(t, d["delta_r"])

        x_lo = max(0.0, t[-1] - PLOT_HISTORY * DT)
        x_hi = t[-1] + 0.5
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(x_lo, x_hi)

        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=PLOT_REFRESH_MS)
    timer.add_callback(_refresh, 0)
    timer.start()

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimNode()

    # ROS spin in a daemon thread — matplotlib MUST own the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        _run_dashboard(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
