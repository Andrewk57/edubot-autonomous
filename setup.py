from setuptools import find_packages, setup

package_name = 'edubot_autonomous'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrew',
    maintainer_email='akazmierc@ltu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
   entry_points={
    'console_scripts': [
        'lane_detection_node = edubot_autonomous.lane_detection_node:main',
        'navigation_node = edubot_autonomous.navigation_node:main',
        'mapping_node = edubot_autonomous.mapping_node:main',
        'cone_detection_node = edubot_autonomous.cone_detection_node:main',
    ],
},
)
