from setuptools import setup, find_packages
PACKAGES = find_packages()

install_requires = ['pandas<=0.22.0', 'python-louvain==0.11', 'numpy',
                    'scipy', 'networkx', 'seaborn', 'nibabel',
                    'forceatlas2', 'pillow']

if __name__ == '__main__':
    setup(
        name='BrainNetworksInPython',
        version='0.1dev',
        packages=PACKAGES,
        package_data={'': ['*.txt', '*.csv']},
        license='MIT license',
        install_requires=install_requires,
        tests_require=['pytest', 'unittest'],
        test_suite='py.test',
    )
