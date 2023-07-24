from setuptools import setup, find_packages
PACKAGES = find_packages()

install_requires = [
   "pandas<1.5",
   "python-louvain==0.11",
   "numpy<1.20.0",
   "matplotlib<3.6",
   "scikit-learn",
   "scipy",
   "networkx>=2.2,<3",
   "seaborn",
   "forceatlas2",
   "nilearn==0.5.2"]


if __name__ == '__main__':
    setup(
        name='scona',
        version='0.1dev',
        packages=PACKAGES,
        package_data={'': ['*.txt', '*.csv']},
        license='MIT license',
        install_requires=install_requires,
        tests_require=['pytest', 'unittest'],
        test_suite='py.test',
    )
