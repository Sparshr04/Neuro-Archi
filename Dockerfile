# Hybrid VTOL Neuro-Fusion — Docker Environment
# ROS 2 Humble (Perception) on Ubuntu 22.04 LTS
# Target: macOS M1/M2/M3 (ARM64)

FROM ros:humble-perception

# ── Install System Dependencies ──────────────────────────────────────────────
# - python3-pip: for ML libraries
# - x11-apps: for X11 forwarding verification
# - ros-humble-rmw-cyclonedds-cpp: better reliable networking for simulation
# - nano/vim: basic editing
RUN apt-get update && apt-get install -y \
    python3-pip \
    x11-apps \
    ros-humble-rmw-cyclonedds-cpp \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python ML/Sim Libraries ──────────────────────────────────────────
# install tensorflow-cpu (ARM64 compatible), mavsdk, matplotlib, scipy.
# Using --break-system-packages because we are in a container environment.
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    tensorflow \
    mavsdk \
    matplotlib \
    scipy 


# ── Workspace Setup ──────────────────────────────────────────────────────────
# We create the workspace structure but rely on volume mounting for src/
WORKDIR /root/ros2_ws
RUN mkdir -p src

# ── Environment Setup ────────────────────────────────────────────────────────
# Source ROS 2 automatically when entering the container
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set collision-free DDS config (optional but recommended for sim)
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Keep container running indefinitely
CMD ["/bin/bash"]
