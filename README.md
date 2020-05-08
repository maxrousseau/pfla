pfla: Python Facial Landmark Analysis
=====================================
[![GitHub release](https://img.shields.io/github/release/maxrousseau/pfla.svg)](https://github.com/maxrousseau/pfla/releases)
[![PyPI license](https://img.shields.io/pypi/l/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI version fury.io](https://badge.fury.io/py/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pfla.svg)](https://pypi.org/project/pfla/)
[![Documentation
Status](https://readthedocs.org/projects/pfla/badge/?version=master)](https://pfla.readthedocs.io/en/master/?badge=master)
[![Build Status](https://travis-ci.org/maxrousseau/pfla.svg?branch=master)](https://travis-ci.org/maxrousseau/pfla)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00855/status.svg)](https://doi.org/10.21105/joss.00855)

A simple command line interface to automate facial analysis. ```pfla``` uses a
pre-trained neural networks to detect faces and annotate them with 68
landmarks. The program also compyte four commonly used facial metrics. The
output is saved to a file to allow for easy statistical analysis by the user.

Publication
-----------

This software was published in
[JOSS](https://joss.theoj.org/papers/10.21105/joss.00855). Since version 1.0.0,
the packaged has changed considerably. The publication release is still
available [here](https://github.com/maxrousseau/pfla/releases/tag/v0.1.1).


Citing
------

If you use this software please use this citation:

```
@article{Rousseau_2018,
doi = {10.21105/joss.00855},
url = {https://doi.org/10.21105%2Fjoss.00855},
year = 2018,
month = {dec},
publisher = {The Open Journal},
volume = {3},
number = {32},
pages = {855},
author = {Maxime Rousseau and Jean-Marc Retrouvey},
title = {pfla: A Python Package for Dental Facial Analysis using Computer Vision and Statistical Shape Analysis},
journal = {Journal of Open Source Software}}
```

Features
--------

- Face detection using mtcnn
- Landmark placement
- Facial metric calculations

Requirements and Dependencies
-----------------------------

-   Python 3.5 (or higher)
-   Python packages:
	* numpy
	* pandas
	* pytest
	* pillow
	* facenet-pytorch
	* face-alignment
	* pytest-cov
	* pytorch

Installation
------------

Install with **pip**:

```shell
pip install -r requirements-pytorch.txt \ # pytorch for CPU
	 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt # other dependencies
pip install pfla
```

Usage
-----


```shell
usage: pfla [-h] [-d] [-l] [-m] [-o OUTPUT] [-v] path

PFLA: python facial landmark analysis. This program will read the image(s)
given as input and can apply a face detection algorithm, landmark placement
and computation of metrics. The results are returned as a text stream.

positional arguments:
  path                  path to the image or directory of images

optional arguments:
  -h, --help            show this help message and exit
  -d, --detect          detect faces and output bounding box
  -l, --landmark        annotate detected faces and output coordinates
  -m, --metrics         compute metrics and output results
  -o OUTPUT, --output OUTPUT
                        specify output filename and format/filetype of the
                        data
  -v, --verbose         increase output verbosity

AUTHOR: Maxime Rousseau LICENSE: MIT
```

Testing
-------

To test your installation run the following commands:

```shell
cd [PATH_TO_PACKAGE_INSTALLATION]
pytest
```
Documentation
-------------

Documentation of the package can be found here:
<https://pfla.readthedocs.io/en/master>

Contribute
----------

-   Contribution guidelines: <https://github.com/maxrousseau/pfla/blob/master/contributing.md>
-   Issue Tracker: <https://github.com/maxrousseau/pfla/issues>
-   Source Code: <https://github.com/maxrousseau/pfla>

License
-------

The project is licensed under the MIT license.

Contact
-------

Maxime Rousseau, DMD candidate 2020 McGill University, Faculty of Dentistry
- Email: <maximerousseau08@gmail.com>
- Website: <https://maxrousseau.github.io/portfolio/>

