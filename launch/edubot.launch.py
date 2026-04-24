from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='edubot_autonomous',
            executable='lane_detection_node',
            name='lane_detection_node',
            output='screen'
        ),
        Node(
            package='edubot_autonomous',
            executable='navigation_node',
            name='navigation_node',
            output='screen'
        ),
        Node(
            package='edubot_autonomous',
            executable='mapping_node',
            name='mapping_node',
            output='screen'
        ),
        Node(
            package='edubot_autonomous',
            executable='cone_detection_node',
            name='cone_detection_node',
            output='screen'
        ),
    ])