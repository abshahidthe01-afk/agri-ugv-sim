from glob import glob

from setuptools import setup

package_name = 'agri_ugv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abdullah Bin Shahid',
    maintainer_email='abshahidthe01@gmail.com',
    description='Four-wheel-steering kinematics and control for the agricultural UGV.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematic_sim = agri_ugv_control.kinematic_sim_node:main',
            'four_ws_driver = agri_ugv_control.four_ws_driver_node:main',
        ],
    },
)
