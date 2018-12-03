from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pfla',
      version='0.2.2',
      description='Python facial landmarking and analysis',
      long_description=readme(),
      classifiers=[
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Image Recognition'
	],
      url='https://github.com/maxrousseau/pfla',
      author='Maxime Rousseau',
      author_email='maximerousseau08@gmail.com',
      include_package_data=True,
      license='MIT',
      packages=find_packages(),
      package_data={'pfla': ['data/haarcascade_frontalface_default.xml',
                             'data/test_females/*.jpg',
                             'data/test_males/*.jpg']},
      zip_safe=False,
      entry_points={
            'console_scripts': ['pfla=pfla.cli:pfla']
      })
