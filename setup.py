from setuptools import setup, find_packages

install_requires = [
   "pandas",
   "python-louvain==0.11",
   "numpy",
   "scikit-learn",
   "scipy",
   "networkx>=2.2",
   "seaborn",
   "forceatlas2",
   "nilearn==0.5.2"]


if __name__ == '__main__':
    setup(
        name='scona',
        version='0.1dev',
        packages=find_packages(),
        package_data={'': ['*.txt', '*.csv']},
        license='MIT license',
        install_requires=install_requires,
        tests_require=['pytest', 'unittest'],
        test_suite='py.test',
        entry_points={
            'console_scripts' : [
                'scona = scona.wrappers.scona:main',
                'corrmat_from_regionalmeasures = scona.wrappers.corrmat_from_regionalmeasures:main',
                'network_analysis_from_corrmat = scona.wrappers.network_analysis_from_corrmat:main']},
        )
