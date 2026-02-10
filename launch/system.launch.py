"""system.launch.py — ROS 2 launch: Neuro-Adaptive Sensor Fusion + SITL Simulator.

Full demo topology
------------------
  [SimNode 400Hz] → /ekf/innovations → [InnovationMonitor] → /neuro/features
                                        → [DNNInference]     → /neuro/covariance_correction
                     ◀────────────────────────────────────────  [CovarianceInjector] (mock log)
                     ◀──── /neuro/covariance_correction ────── (also feeds back into SimNode)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Arguments ────────────────────────────────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to the INT8 TFLite model.",
    )
    alpha_arg = DeclareLaunchArgument(
        "alpha",
        default_value="1.0",
        description="Blending gain α (Eq 30): R_adapt = R₀ + α·ΔR.",
    )
    window_size_arg = DeclareLaunchArgument(
        "window_size",
        default_value="64",
        description="Innovation sliding-window length W.",
    )
    decimation_arg = DeclareLaunchArgument(
        "decimation",
        default_value="8",
        description="Feature extraction decimation factor.",
    )

    # ── Nodes ────────────────────────────────────────────────────────────────
    sim_node = Node(
        package="neuro_adaptive_fusion",
        executable="sim_node",
        name="sim_node",
        output="screen",
    )

    innovation_monitor = Node(
        package="neuro_adaptive_fusion",
        executable="innovation_monitor",
        name="innovation_monitor",
        output="screen",
        parameters=[
            {
                "window_size": LaunchConfiguration("window_size"),
                "n_channels": 6,
                "decimation": LaunchConfiguration("decimation"),
            }
        ],
    )

    dnn_inference = Node(
        package="neuro_adaptive_fusion",
        executable="dnn_inference",
        name="dnn_inference",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "alpha": LaunchConfiguration("alpha"),
            }
        ],
    )

    covariance_injector = Node(
        package="neuro_adaptive_fusion",
        executable="covariance_injector",
        name="covariance_injector",
        output="screen",
    )

    return LaunchDescription(
        [
            model_path_arg,
            alpha_arg,
            window_size_arg,
            decimation_arg,
            LogInfo(msg="🚀  Launching Neuro-Adaptive Fusion + SITL Simulator …"),
            sim_node,
            innovation_monitor,
            dnn_inference,
            covariance_injector,
        ]
    )
