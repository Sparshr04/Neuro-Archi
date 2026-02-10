"""train_model.py — Train the Supervisor DNN (Section 3.3).

Architecture (paper spec):
    Input   ξ_j ∈ ℝ^{4m}   (m=6, so 24 features)
    Dense(64, ReLU)
    Dense(32, ReLU)
    Dense(16, ReLU)
    Dense(m,  ReLU)          → ΔR ≥ 0  (non-negative covariance correction)

Loss:  MSE( ΔR_pred, ΔR_true )

Usage
-----
    python -m ml_core.train_model
    python -m ml_core.train_model --data-dir ml_core/data --epochs 120 --batch-size 128
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import tensorflow as tf


# ── Defaults ─────────────────────────────────────────────────────────────────
M: int = 6  # measurement dimension (6-DOF IMU)
N_FEATURES: int = 4 * M  # 24  (mean, var, NIS, ZCR per channel)
BATCH_SIZE: int = 128
EPOCHS: int = 100
LEARNING_RATE: float = 1e-3
VALIDATION_SPLIT: float = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════


def build_supervisor_dnn(input_dim: int = N_FEATURES, output_dim: int = M) -> tf.keras.Model:
    """Build the Supervisor DNN as specified in Section 3.3.

    Parameters
    ----------
    input_dim : int
        Feature vector length (4 × m).
    output_dim : int
        Number of diagonal ΔR elements to predict (m).

    Returns
    -------
    tf.keras.Sequential
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="feature_vector"),
            tf.keras.layers.Dense(64, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(32, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(16, activation="relu", name="hidden_3"),
            tf.keras.layers.Dense(output_dim, activation="relu", name="delta_R"),
        ],
        name="SupervisorDNN",
    )

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_data(
    data_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features (X) and labels (Y) from .npy files.

    Returns
    -------
    X : (N, 24) float32 — feature vectors  ξ_j
    Y : (N, 6)  float32 — ground-truth ΔR = R* − R₀
    """
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            "Run `python -m ml_core.generate_data` first."
        )

    X = np.load(features_path)
    Y = np.load(labels_path)

    # ΔR must be non-negative (clamp any numerical noise from label gen)
    Y = np.clip(Y, 0.0, None)

    print(f"[train_model] Loaded data from {data_dir}")
    print(f"  X: {X.shape}  dtype={X.dtype}")
    print(f"  Y: {Y.shape}  dtype={Y.dtype}")
    print(f"  Y range: [{Y.min():.6f}, {Y.max():.6f}]")

    return X, Y


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    data_dir: pathlib.Path = pathlib.Path("ml_core/data"),
    output_dir: pathlib.Path = pathlib.Path("ml_core/models"),
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    val_split: float = VALIDATION_SPLIT,
) -> None:
    """Full training loop with callbacks and SavedModel export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X, Y = load_data(data_dir)

    # ── Shuffle before split ─────────────────────────────────────────────────
    n = len(X)
    idx = np.random.default_rng(42).permutation(n)
    X, Y = X[idx], Y[idx]

    split = int(n * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"\n  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}\n")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_supervisor_dnn(input_dim=X.shape[1], output_dim=Y.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    # Keras 3: .save() only supports .keras; use .export() for SavedModel
    saved_model_path = output_dir / "supervisor_dnn"
    model.export(saved_model_path)
    print(f"\n[train_model] SavedModel  → {saved_model_path}")

    keras_path = output_dir / "supervisor_dnn.keras"
    model.save(keras_path)
    print(f"[train_model] Keras model → {keras_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    best_epoch_idx = int(np.argmin(history.history["val_loss"]))
    best_val_loss = history.history["val_loss"][best_epoch_idx]
    best_val_mae = history.history["val_mae"][best_epoch_idx]

    print(f"\n{'═' * 55}")
    print(f"  Best epoch:    {best_epoch_idx + 1}")
    print(f"  Val MSE:       {best_val_loss:.8f}")
    print(f"  Val MAE:       {best_val_mae:.8f}")
    print(f"  Parameters:    {model.count_params():,}")
    print(f"{'═' * 55}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Supervisor DNN for neuro-adaptive EKF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml_core/data",
        help="Directory containing features.npy and labels.npy.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="ml_core/models", help="Directory to save trained model."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--val-split", type=float, default=VALIDATION_SPLIT)
    args = parser.parse_args()

    train(
        data_dir=pathlib.Path(args.data_dir),
        output_dir=pathlib.Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )
