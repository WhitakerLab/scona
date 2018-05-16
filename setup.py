from setuptools import setup, find_packages
PACKAGES = find_packages()
dir_datasets = 'BrainNetworksInPython/datasets'
DATA = [(dir_datasets, glob(".txt"),
        (dir_datasets, glob(".csv")]

if __name__ == '__main__':
    setup(
    name='BrainNetworksInPython',
    version='0.1dev',
    packages=PACKAGES,
    data_files=DATA
    license='MIT license',
)
