"""generate_data.py — Synthetic Innovation Sequence Generator for Neuro-Adaptive EKF.

Generates training data that mimics the EKF innovation sequence ỹ_k of a
Hybrid VTOL aircraft across three flight phases:

    Hover (Gaussian)  →  Transition Corridor (Non-Gaussian)  →  Cruise (Gaussian)

The Transition Corridor injects:
  • Heavy-tailed (Student-t) noise — models aerodynamic buffeting
  • Time-varying sinusoidal vibration η_vib(t) — models rotor / wing coupling
  • Amplitude ramp envelope — models progressive flow separation

Feature extraction follows Equations 27–29 of the Springer chapter:
  1. Sample Mean   μ̂_W           over sliding window W
  2. Sample Variance σ̂²_W
  3. Normalised Innovation Squared (NIS)
  4. Zero-Crossing Rate (ZCR)

Labels:  ΔR_k = R*_k − R₀   (true injected covariance minus baseline).

Usage
-----
    python -m ml_core.generate_data --n-flights 50 --duration 120
    python -m ml_core.generate_data --help
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SIZE: int = 64  # W — sliding window for feature extraction
N_CHANNELS: int = 6  # ax, ay, az, gx, gy, gz innovation channels
IMU_RATE_HZ: int = 200  # EKF innovation output rate

# Baseline measurement-noise std-devs (√R₀ diagonal), ICM-42688-P datasheet
R0_STD = np.array(
    [
        0.012,  # accel x  (m/s²)
        0.012,  # accel y
        0.012,  # accel z
        0.004,  # gyro  x  (rad/s)
        0.004,  # gyro  y
        0.004,  # gyro  z
    ],
    dtype=np.float64,
)

# R₀ diagonal (variance)
R0_VAR = R0_STD**2  # (6,)


# ═══════════════════════════════════════════════════════════════════════════════
# Flight-phase configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class PhaseConfig:
    """Timing ratios for a single flight profile (fractions of total duration)."""

    hover_ratio: float = 0.30
    transition_ratio: float = 0.40
    cruise_ratio: float = 0.30

    def __post_init__(self) -> None:
        total = self.hover_ratio + self.transition_ratio + self.cruise_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Phase ratios must sum to 1.0, got {total:.4f}")


@dataclasses.dataclass(frozen=True)
class TransitionParams:
    """Physics parameters for the transition-corridor noise model."""

    # Student-t degrees of freedom (lower → heavier tails → more extreme outliers)
    student_t_df: float = 4.0
    # Peak vibration amplitude scale (multiple of R₀ std)
    vib_amplitude_scale: float = 8.0
    # Vibration frequency band [Hz] — rotor / wing aero coupling
    vib_freq_low: float = 12.0
    vib_freq_high: float = 45.0
    # Number of superimposed vibration harmonics
    n_harmonics: int = 5
    # Peak noise variance multiplier over R₀ at the centre of transition
    peak_var_multiplier: float = 15.0


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Class
# ═══════════════════════════════════════════════════════════════════════════════


class InnovationSequenceSimulator:
    """Simulate an EKF innovation sequence ỹ_k for a full flight profile.

    The innovation is modelled as:

        ỹ_k = η_base(k) + η_vib(k)

    where η_base ~ N(0, R₀) during hover/cruise, and during the transition
    corridor it becomes a mixture of:
      • Heavy-tailed Student-t component   (aerodynamic buffeting)
      • Oscillatory vibration component    (rotor/wing coupling harmonics)

    both scaled by a smooth ramp envelope that peaks at the corridor centre.

    Parameters
    ----------
    duration_s : float
        Total flight duration in seconds.
    rate_hz : int
        Innovation sample rate (matches EKF output rate).
    phase_cfg : PhaseConfig
        Hover / transition / cruise timing split.
    trans_params : TransitionParams
        Transition-corridor noise physics.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        duration_s: float = 60.0,
        rate_hz: int = IMU_RATE_HZ,
        phase_cfg: PhaseConfig | None = None,
        trans_params: TransitionParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.duration_s = duration_s
        self.rate_hz = rate_hz
        self.n_samples = int(duration_s * rate_hz)
        self.dt = 1.0 / rate_hz
        self.phase_cfg = phase_cfg or PhaseConfig()
        self.tp = trans_params or TransitionParams()
        self.rng = np.random.default_rng(seed)

        # Pre-compute phase boundaries (sample indices)
        n = self.n_samples
        self._hover_end = int(n * self.phase_cfg.hover_ratio)
        self._trans_end = int(n * (self.phase_cfg.hover_ratio + self.phase_cfg.transition_ratio))

        # Timestamps
        self.timestamps = np.arange(n) * self.dt

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self) -> dict[str, np.ndarray]:
        """Run the full simulation.

        Returns
        -------
        dict with keys:
            "innovations"   : (N, 6) — synthetic innovation sequence ỹ_k
            "true_R_diag"   : (N, 6) — instantaneous true noise variance R*_k
            "phase_labels"  : (N,)   — 0=hover, 1=transition, 2=cruise
            "timestamps"    : (N,)
        """
        n, ch = self.n_samples, N_CHANNELS

        innovations = np.zeros((n, ch), dtype=np.float64)
        true_R_diag = np.zeros((n, ch), dtype=np.float64)
        phase_labels = np.zeros(n, dtype=np.int8)

        # ── Hover phase ──────────────────────────────────────────────────────
        h_end = self._hover_end
        innovations[:h_end] = self._gaussian_phase(h_end)
        true_R_diag[:h_end] = R0_VAR[np.newaxis, :]
        phase_labels[:h_end] = 0

        # ── Transition corridor ──────────────────────────────────────────────
        t_start, t_end = h_end, self._trans_end
        t_len = t_end - t_start
        innov_trans, R_trans = self._transition_phase(t_start, t_len)
        innovations[t_start:t_end] = innov_trans
        true_R_diag[t_start:t_end] = R_trans
        phase_labels[t_start:t_end] = 1

        # ── Cruise phase ─────────────────────────────────────────────────────
        c_len = n - t_end
        innovations[t_end:] = self._gaussian_phase(c_len)
        true_R_diag[t_end:] = R0_VAR[np.newaxis, :]
        phase_labels[t_end:] = 2

        return {
            "innovations": innovations,
            "true_R_diag": true_R_diag,
            "phase_labels": phase_labels,
            "timestamps": self.timestamps,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _gaussian_phase(self, length: int) -> np.ndarray:
        """Nominal Gaussian noise: ỹ_k ~ N(0, R₀)."""
        return self.rng.normal(0.0, R0_STD[np.newaxis, :], size=(length, N_CHANNELS))

    def _transition_phase(self, start_idx: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Non-Gaussian, vibration-laden innovation during the transition corridor.

        Noise model:
            ỹ_k = envelope(k) · [ t_noise(k) + η_vib(k) ]

        where:
          • envelope(k) is a smooth (raised-cosine) ramp peaking at corridor centre
          • t_noise    ~ t_ν(0, R₀)  (Student-t, heavy tails)
          • η_vib(k)   = Σ_h A_h · sin(2π f_h t + φ_h)  (vibration harmonics)

        The true instantaneous variance R*_k = envelope² · σ²_effective.
        """
        tp = self.tp
        t = self.timestamps[start_idx : start_idx + length]  # local time vector

        # ── 1. Smooth ramp envelope (raised cosine, peak at centre) ──────────
        #  envelope ∈ [0, 1], peaks at corridor midpoint
        phase_frac = np.linspace(0, np.pi, length)
        envelope = np.sin(phase_frac)  # smooth 0→1→0

        # ── 2. Student-t component (heavy-tailed base noise) ────────────────
        #  scipy-free: use the identity  t_ν = Z / √(V/ν)
        #  where Z ~ N(0,1), V ~ χ²(ν)  (= Gamma(ν/2, 2))
        z = self.rng.standard_normal((length, N_CHANNELS))
        chi2 = self.rng.gamma(tp.student_t_df / 2.0, 2.0, size=(length, 1))
        t_noise = z / np.sqrt(chi2 / tp.student_t_df)
        # Scale to baseline std
        t_noise *= R0_STD[np.newaxis, :]

        # ── 3. Oscillatory vibration harmonics η_vib(t) ─────────────────────
        #  Superposition of n_harmonics sinusoids in [f_low, f_high] Hz
        freqs = np.linspace(tp.vib_freq_low, tp.vib_freq_high, tp.n_harmonics)
        phases = self.rng.uniform(0, 2 * np.pi, size=(tp.n_harmonics, N_CHANNELS))
        amplitudes = self.rng.uniform(0.5, 1.0, size=(tp.n_harmonics, N_CHANNELS))
        # Normalise amplitudes so peak ≈ vib_amplitude_scale × R₀_std
        amplitudes *= (tp.vib_amplitude_scale * R0_STD[np.newaxis, :]) / tp.n_harmonics

        # Vectorised: (length, 1) × (1, n_harmonics) → (length, n_harmonics)
        # then broadcast across channels
        vib = np.zeros((length, N_CHANNELS), dtype=np.float64)
        for h in range(tp.n_harmonics):
            # (length,) outer product with (channels,)
            vib += amplitudes[h] * np.sin(2 * np.pi * freqs[h] * t[:, np.newaxis] + phases[h])

        # ── 4. Combine with envelope ────────────────────────────────────────
        env_2d = envelope[:, np.newaxis]  # (length, 1)
        # Scale factor ramps from 1.0 (edges) to peak_var_multiplier (centre)
        var_scale = 1.0 + (tp.peak_var_multiplier - 1.0) * envelope
        var_scale_2d = var_scale[:, np.newaxis]  # (length, 1)

        innovations = np.sqrt(var_scale_2d) * t_noise + env_2d * vib

        # ── 5. True instantaneous variance R*_k ─────────────────────────────
        # R*_k ≈ var_scale · R₀ + envelope² · vib_power
        vib_power = np.mean(amplitudes**2, axis=0)  # (6,) average harmonic power
        true_R = var_scale_2d * R0_VAR[np.newaxis, :] + (env_2d**2) * vib_power[np.newaxis, :]

        return innovations, true_R


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction  (Eq 27–29, fully vectorised)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_features(
    innovations: np.ndarray,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """Compute 4 features over a sliding window using stride tricks — zero copies.

    Parameters
    ----------
    innovations : (N, C) array
        Raw innovation sequence, C channels.
    window : int
        Sliding window length W.

    Returns
    -------
    features : (N - W + 1, C * 4) array
        Per-window features in column order:
        [μ̂_ch0, ..., μ̂_chC, σ̂²_ch0, ..., σ̂²_chC, NIS_ch0, ..., ZCR_chC]
    """
    N, C = innovations.shape
    if N < window:
        raise ValueError(f"Sequence length {N} < window {window}")

    # ── Build sliding-window view: (n_windows, W, C) — zero-copy ────────────
    n_win = N - window + 1
    strides = (innovations.strides[0], innovations.strides[0], innovations.strides[1])
    windows = np.lib.stride_tricks.as_strided(
        innovations,
        shape=(n_win, window, C),
        strides=strides,
    )

    # ── 1. Sample Mean  μ̂_W = (1/W) Σ ỹ_k ──────────────────────────────────
    mu = np.mean(windows, axis=1)  # (n_win, C)

    # ── 2. Sample Variance  σ̂²_W = (1/(W-1)) Σ (ỹ_k − μ̂)² ────────────────
    var = np.var(windows, axis=1, ddof=1)  # (n_win, C)

    # ── 3. Normalised Innovation Squared  NIS_k = ỹ_k^T R₀⁻¹ ỹ_k ──────────
    #  Per-window average NIS (scalar per channel for diagonal R₀):
    #    NIS_ch = (1/W) Σ_{k∈W} (ỹ_k,ch)² / R₀_ch
    R0_inv = 1.0 / R0_VAR[np.newaxis, np.newaxis, :]  # (1, 1, C)
    nis = np.mean(windows**2 * R0_inv, axis=1)  # (n_win, C)

    # ── 4. Zero-Crossing Rate  ZCR = (1/(W-1)) Σ |sign(ỹ_k) − sign(ỹ_{k-1})| / 2
    signs = np.sign(windows)  # (n_win, W, C)
    crossings = np.abs(np.diff(signs, axis=1))  # (n_win, W-1, C)
    zcr = np.mean(crossings, axis=1) / 2.0  # (n_win, C), normalised to [0, 1]

    # ── Stack: (n_win, C*4) ─────────────────────────────────────────────────
    features = np.concatenate([mu, var, nis, zcr], axis=1).astype(np.float32)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Label Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_labels(
    true_R_diag: np.ndarray,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """Compute per-window labels  ΔR_k = R*_k − R₀.

    We average R*_k over each window to produce a smooth regression target.

    Parameters
    ----------
    true_R_diag : (N, C) array
        Instantaneous true noise variance at each timestep.

    Returns
    -------
    labels : (N - W + 1, C) float32 array
        Average covariance excess over each window.
    """
    N, C = true_R_diag.shape
    n_win = N - window + 1

    # Sliding-window view of true_R
    strides = (true_R_diag.strides[0], true_R_diag.strides[0], true_R_diag.strides[1])
    R_windows = np.lib.stride_tricks.as_strided(
        true_R_diag,
        shape=(n_win, window, C),
        strides=strides,
    )

    # ΔR = mean(R*) − R₀  over each window
    labels = np.mean(R_windows, axis=1) - R0_VAR[np.newaxis, :]
    return labels.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_dataset(
    n_flights: int = 50,
    duration_s: float = 60.0,
    rate_hz: int = IMU_RATE_HZ,
    window: int = WINDOW_SIZE,
    output_dir: pathlib.Path = pathlib.Path("ml_core/data"),
    seed: int = 42,
) -> None:
    """Generate a full training dataset from multiple simulated flights.

    For each flight, we:
      1. Randomise the transition timing slightly (±10 %)
      2. Generate the innovation sequence
      3. Extract features (X) and labels (Y)
      4. Concatenate across flights

    Output
    ------
    {output_dir}/features.npy   — (M, C*4) float32
    {output_dir}/labels.npy     — (M, C)   float32
    {output_dir}/metadata.npy   — dict with generation parameters
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    t0 = time.perf_counter()

    for i in range(n_flights):
        # Randomise phase ratios slightly for diversity
        jitter = rng.uniform(-0.05, 0.05, size=3)
        base = np.array([0.30, 0.40, 0.30]) + jitter
        base = np.clip(base, 0.10, 0.60)
        base /= base.sum()  # re-normalise

        phase_cfg = PhaseConfig(
            hover_ratio=float(base[0]),
            transition_ratio=float(base[1]),
            cruise_ratio=float(base[2]),
        )

        # Randomise transition severity
        trans_params = TransitionParams(
            student_t_df=rng.uniform(3.0, 6.0),
            vib_amplitude_scale=rng.uniform(5.0, 12.0),
            vib_freq_low=rng.uniform(8.0, 15.0),
            vib_freq_high=rng.uniform(35.0, 55.0),
            n_harmonics=rng.integers(3, 8),
            peak_var_multiplier=rng.uniform(8.0, 25.0),
        )

        sim = InnovationSequenceSimulator(
            duration_s=duration_s,
            rate_hz=rate_hz,
            phase_cfg=phase_cfg,
            trans_params=trans_params,
            seed=seed + i,
        )
        result = sim.generate()

        features = extract_features(result["innovations"], window=window)
        labels = generate_labels(result["true_R_diag"], window=window)

        all_features.append(features)
        all_labels.append(labels)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  Flight {i + 1:>3d}/{n_flights}  |  "
                f"windows: {features.shape[0]:>6,}  |  "
                f"elapsed: {elapsed:.1f}s"
            )

    # ── Concatenate & save ───────────────────────────────────────────────────
    X = np.concatenate(all_features, axis=0)  # (M, C*4)
    Y = np.concatenate(all_labels, axis=0)  # (M, C)

    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", Y)

    metadata = {
        "n_flights": n_flights,
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "window": window,
        "n_channels": N_CHANNELS,
        "n_features_per_channel": 4,
        "feature_order": ["mean", "variance", "nis", "zcr"],
        "X_shape": X.shape,
        "Y_shape": Y.shape,
        "seed": seed,
        "R0_var": R0_VAR.tolist(),
    }
    np.save(output_dir / "metadata.npy", metadata)

    total_time = time.perf_counter() - t0
    print(f"\n{'═' * 65}")
    print(f"  Dataset generated in {total_time:.1f}s")
    print(f"  Features (X):  {X.shape}  →  {output_dir / 'features.npy'}")
    print(f"  Labels   (Y):  {Y.shape}  →  {output_dir / 'labels.npy'}")
    print(f"  Size:  {(X.nbytes + Y.nbytes) / 1024 / 1024:.1f} MB")
    print(f"{'═' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic innovation sequences for DNN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-flights", type=int, default=50, help="Number of simulated flight profiles."
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Duration of each flight in seconds."
    )
    parser.add_argument(
        "--rate-hz", type=int, default=IMU_RATE_HZ, help="Innovation sample rate (Hz)."
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_SIZE, help="Feature extraction window size W."
    )
    parser.add_argument(
        "--output-dir", type=str, default="ml_core/data", help="Output directory for .npy files."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    build_dataset(
        n_flights=args.n_flights,
        duration_s=args.duration,
        rate_hz=args.rate_hz,
        window=args.window,
        output_dir=pathlib.Path(args.output_dir),
        seed=args.seed,
    )
