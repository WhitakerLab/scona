from setuptools import setup, find_packages
PACKAGES = find_packages()

install_requires = [
   "pandas",
   "python-louvain==0.11",
   "numpy",
   "scikit-learn",
   "scipy",
   "networkx>=2.2",
   "seaborn",
   "forceatlas2",
   "nilearn==0.5.0a0"]


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
        scripts=[
            'wrappers/corrmat_from_regionalmeasures',
            'wrappers/network_analysis_from_corrmat',
            'wrappers/scona'],
    )
