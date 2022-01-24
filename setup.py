from distutils.core import setup

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
      python_requires='>3.6'
      )
