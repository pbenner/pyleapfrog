import numpy as np

from setuptools   import setup, Extension
from Cython.Build import cythonize

setup(name='leapfrog',
      version='0.0.1',
      description='Leapfrog regularization for PyTorch',
      long_description='file: README.md',
      url='https://github.com/pbenner/pyleapfrog',
      author='Philipp Benner',
      author_email='philipp.benner@gmail.com',
      license='MIT',
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      packages=['leapfrog'],
      install_requires=['torch', 'numpy'],
      python_requires='>3.6',
      ext_modules = cythonize([
          Extension('leapfrog_util', sources=["leapfrog/leapfrog_util.pyx"], include_dirs=[np.get_include()], define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
      ], compiler_directives={'language_level' : "3"})
      )
