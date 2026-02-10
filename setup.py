from setuptools import find_packages, setup
from glob import glob
import os

package_name = "neuro_adaptive_fusion"

setup(
    name=package_name,
    version="0.1.0",
    # Include both the main ROS package and our ML core library
    packages=[package_name, "ml_core"],
    package_dir={
        package_name: "neuro_adaptive_fusion",
        "ml_core": "ml_core",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # Include TFLite models for runtime
        (os.path.join("share", package_name, "models"), glob("ml_core/models/*.tflite")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sparsh",
    maintainer_email="sparsh@todo.com",
    description="Neuro-Adaptive Sensor Fusion ROS 2 nodes for Hybrid VTOL UAV.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "innovation_monitor = neuro_adaptive_fusion.innovation_monitor:main",
            "dnn_inference = neuro_adaptive_fusion.dnn_inference:main",
            "covariance_injector = neuro_adaptive_fusion.covariance_injector:main",
            "sim_node = neuro_adaptive_fusion.sim_node:main",
        ],
    },
)
