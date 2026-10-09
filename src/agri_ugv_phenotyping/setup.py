from glob import glob

from setuptools import setup

package_name = 'agri_ugv_phenotyping'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abdullah Bin Shahid',
    maintainer_email='abshahidthe01@gmail.com',
    description='Phenotyping: a plant map of every plot from the laser line scanners.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'canopy_truth = agri_ugv_phenotyping.truth:main',
            'plant_map = agri_ugv_phenotyping.plant_map_node:main',
        ],
    },
)
