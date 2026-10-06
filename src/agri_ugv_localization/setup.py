from setuptools import setup

package_name = 'agri_ugv_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abdullah Bin Shahid',
    maintainer_email='abshahidthe01@gmail.com',
    description='Localization: simulated RTK GNSS errors and the dual-antenna heading.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gnss_errors = agri_ugv_localization.gnss_node:main',
            'localization = agri_ugv_localization.localization_node:main',
        ],
    },
)
