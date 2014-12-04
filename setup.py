from setuptools import setup
import sys


if sys.version_info.major > 3:
    print("This was created with python 2 in mind, though porting should be easy")
    sys.exit(1)

setup(
    name='wtmm-python',
    version='0.1',
    packages=['wtmm'],
    url='https://github.com/buckie/wtmm-python',
    license='BSD',
    author='buckie',
    author_email='wjmartino@gmail.com',
    description='Implementation of the Wavelet Transform Modulus Maxima method of '
                'feature abstraction and local Holder exponent estimation ',
    install_requires=[
        'numpy>=1.8.1',
        'matplotlib>=1.3.1',
        'scipy>=0.14'
    ],
)

