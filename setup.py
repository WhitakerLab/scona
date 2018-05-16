from setuptools import setup, find_packages
PACKAGES = find_packages()

if __name__ == '__main__':
    setup(
    name='BrainNetworksInPython',
    version='0.1dev',
    packages=PACKAGES,
    package_data={'': ['.txt','.csv']},
    license='MIT license',
)
