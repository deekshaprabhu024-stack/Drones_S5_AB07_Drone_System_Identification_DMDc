import os
from glob import glob
from setuptools import find_packages, setup

package_name = "quad_dmdc_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models", "x500_dmdc"),
         glob("models/x500_dmdc/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Drone system identification using DMDc on Gazebo (gz-sim)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "data_collection = quad_dmdc_sim.data_collection:main",
            "run_identification = quad_dmdc_sim.run_identification:main",
            "fly_manual = quad_dmdc_sim.simulate:main",
        ],
    },
)
