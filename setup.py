from distutils.core import setup

from setuptools import find_packages
PACKAGES = find_packages()

if __name__ == '__main__':
    setup(
    name='BrainNetworksInPython',
    version='0.1dev',
    packages=PACKAGES,
    license='MIT license',
)
