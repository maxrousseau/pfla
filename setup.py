from setuptools import setup, find_packages

setup(name='pfla',
      version='1.0.0',
      description='Python facial landmarking and analysis',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      classifiers=['Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Image Recognition'],
      url='https://github.com/maxrousseau/pfla',
      project_urls={
        'Contribution guidelines':'https://github.com/maxrousseau/pfla/blob/master/contributing.md',
        'Issue Tracker': 'https://github.com/maxrousseau/pfla/issues',
        'Source Code': 'https://github.com/maxrousseau/pfla',

      },
      author='Maxime Rousseau',
      author_email='maximerousseau08@gmail.com',
      include_package_data=True,
      license='MIT',
      packages=find_packages(),
      package_data={'pfla': ['/test/data/*.jpg']},
      zip_safe=False,
      entry_points={'console_scripts': ['pfla=pfla.cli:pfla']}
)
