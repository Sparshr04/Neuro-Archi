# Hybrid VTOL — Neuro-Adaptive Sensor Fusion

Research-grade implementation of a **Neuro-Adaptive EKF** for a Hybrid VTOL aircraft.  
Companion Computer (Raspberry Pi 5) ↔ Flight Controller (Cube Orange+).

---

## Architecture

```
Cube Orange+ (PX4 EKF2)              Raspberry Pi 5
────────────────────────              ──────────────
  EKF innovations ── MAVLink ──▶  InnovationMonitor
                                       │  deque(W=64), decimate every 8th
                                  /neuro/features (24-dim)
                                       │
                                  DNNInference
                                       │  neuro_adapter.tflite (INT8, 10 KB)
                                       │  Eq 30: R_adapt = R₀ + α · ΔR
                                  /neuro/covariance_correction
                                       │
                                  CovarianceInjector
                                       │
  EKF2 R matrix ◀── MAVLink ──────────┘
```

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| [uv](https://docs.astral.sh/uv/) | latest |
| ROS 2 | Humble |
| TensorFlow | ≥ 2.16 |

---

## 1 — Install Dependencies

```bash
cd hybrid_vtol_neuro_fusion

# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install project + dev tools
uv pip install -e ".[dev,viz]"

# Install TensorFlow (training, host only)
uv pip install tensorflow
```

## 2 — Generate Synthetic Training Data

```bash
python -m ml_core.generate_data \
    --n-flights 50 \
    --duration 120 \
    --seed 42
```

Output: `ml_core/data/features.npy` and `ml_core/data/labels.npy`.

## 3 — Train the Supervisor DNN

```bash
python -m ml_core.train_model \
    --data-dir ml_core/data \
    --epochs 100 \
    --batch-size 128
```

Output: `ml_core/models/supervisor_dnn/` (SavedModel) and `.keras`.

## 4 — Quantize to INT8 TFLite

```bash
python -m ml_core.quantize \
    --model ml_core/models/supervisor_dnn \
    --data-dir ml_core/data \
    --output ml_core/models/neuro_adapter.tflite
```

Output: `ml_core/models/neuro_adapter.tflite` (~10 KB, full INT8).

## 5 — Launch the ROS 2 Stack

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build the workspace
cd src/neuro_adaptive_fusion
colcon build --packages-select neuro_adaptive_fusion
source install/setup.bash

# Launch all 3 nodes
ros2 launch neuro_adaptive_fusion system.launch.py \
    model_path:=ml_core/models/neuro_adapter.tflite \
    alpha:=1.0
```

## Project Structure

```
hybrid_vtol_neuro_fusion/
├── pyproject.toml
├── ml_core/
│   ├── generate_data.py          # Synthetic innovation sequences
│   ├── train_model.py            # Supervisor DNN training
│   ├── quantize.py               # INT8 TFLite quantization
│   └── models/                   # Saved artifacts
├── src/neuro_adaptive_fusion/
│   ├── package.xml
│   ├── setup.py
│   └── neuro_adaptive_fusion/
│       ├── innovation_monitor.py # Sliding-window feature extraction
│       ├── dnn_inference.py      # TFLite inference + Eq 30
│       ├── covariance_injector.py# Mock MAVLink injection
│       └── utils/
└── launch/
    └── system.launch.py
```

## Hardware

| Component | Role | Interface |
|-----------|------|-----------|
| Cube Orange+ | Flight Controller | MAVLink (UART) |
| Raspberry Pi 5 | Companion Computer | USB / UART |
| u-blox NEO-M9N | GPS Receiver | I²C |
| ICM-42688-P | IMU (on Cube) | SPI (internal) |

---

## License

MIT
