from setuptools import find_packages, setup
import os
from glob import glob

package_name = "neuro_adaptive_fusion"

setup(
    name=package_name,
    version="0.1.0",
    # find_packages() correctly detects both 'neuro_adaptive_fusion' and 'ml_core'
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/system.launch.py"]),
        # CRITICAL: Re-added this line so the node can find your AI model
        (
            os.path.join("share", package_name, "models"),
            glob("ml_core/models/*.tflite"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sparsh",
    maintainer_email="sparshr2004@gmail.com",
    description="Neuro-Adaptive Sensor Fusion Package",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "innovation_monitor = neuro_adaptive_fusion.innovation_monitor:main",
            "dnn_inference = neuro_adaptive_fusion.dnn_inference:main",
            "covariance_injector = neuro_adaptive_fusion.covariance_injector:main",
            "sim_node = neuro_adaptive_fusion.sim_node:main",
        ],
    },
)
