from setuptools import setup, find_packages


setup(
    name='bwai_examples',
    author='bw',
    version='0.0.1',
    description='Real examples of bwai',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'matplotlib',
        # 'bwai',
    ],
)
