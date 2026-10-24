"""Launch Gazebo Harmonic and the ROS-Gazebo bridge for the DMDc quadrotor."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("quad_dmdc_sim")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    world_path = os.path.join(pkg_share, "worlds", "dmdc_world.sdf")
    bridge_config = os.path.join(pkg_share, "config", "bridge.yaml")
    model_path = os.path.join(pkg_share, "models")

    gui_arg = DeclareLaunchArgument(
        "gui", default_value="true",
        description="Show Gazebo GUI; use false for headless data collection",
    )

    # Gazebo resolves model:// URIs through GZ_SIM_RESOURCE_PATH.
    resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=model_path + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    gz_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    # GUI mode: normal Gazebo server + GUI.
    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=IfCondition(LaunchConfiguration("gui")),
        launch_arguments={
            "gz_args": f"-r {world_path}",
            "on_exit_shutdown": "True",
        }.items(),
    )

    # Headless mode: server only.
    gz_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=UnlessCondition(LaunchConfiguration("gui")),
        launch_arguments={
            "gz_args": f"-s -r {world_path}",
            "on_exit_shutdown": "True",
        }.items(),
    )

    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="dmdc_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
    )

    return LaunchDescription([
        gui_arg,
        resource_path,
        gz_gui,
        gz_headless,
        bridge_node,
    ])
