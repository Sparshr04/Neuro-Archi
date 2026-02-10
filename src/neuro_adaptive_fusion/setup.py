from setuptools import find_packages, setup

package_name = "neuro_adaptive_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["../../launch/system.launch.py"]),
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
