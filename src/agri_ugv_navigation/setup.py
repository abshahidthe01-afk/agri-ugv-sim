from glob import glob

from setuptools import setup

package_name = 'agri_ugv_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abdullah Bin Shahid',
    maintainer_email='abshahidthe01@gmail.com',
    description='Navigation: mission planning over the field and segment following.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission = agri_ugv_navigation.mission_node:main',
        ],
    },
)
