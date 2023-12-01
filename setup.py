from setuptools import find_packages, setup

setup(
    name='compr_ch_ess',
    packages=find_packages(include=['compr_ch_ess']),
    version='0.1.0',
    description='Adaptive Chess PGN Compressor',
    author='Tyler Lin',
    license='MIT',
    install_requires=['pandas', 'numpy', 'chess', 'tensorflow>=2.0.0'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
