# Third Party
from setuptools import setup

setup(
   name='sparselib',
   version='1.0.0',
   description='Sparse spectral discretization for linear advection',
   author='Parker Ewen',
   author_email='parker.ewen5441@gmail.com',
   packages=['sparselib'],
   install_requires=['matplotlib', 
                     'numpy', 
                     'scipy', 
                     'tqdm'],
)
