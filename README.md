pfla: Python Facial Landmark Analysis
=====================================
[![GitHub release](https://img.shields.io/github/release/maxrousseau/pfla.svg)](https://github.com/maxrousseau/pfla/releases)
[![PyPI license](https://img.shields.io/pypi/l/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI version fury.io](https://badge.fury.io/py/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pfla.svg)](https://pypi.org/project/pfla/)
[![Read the Docs](https://img.shields.io/readthedocs/pip.svg)](https://pfla.readthedocs.io/en/latest/index.html#)
[![Build Status](https://travis-ci.org/maxrousseau/pfla.svg?branch=master)](https://travis-ci.org/maxrousseau/pfla)
[![JOSS](http://joss.theoj.org/papers/d86beb0eb37afd606630b2535e88c4a2/status.svg)](http://joss.theoj.org/papers/d86beb0eb37afd606630b2535e88c4a2)


Advances in artificial intelligence have enhanced the usability of these
technologies in a clinical setting. This python package introduces the use of a
Detection Outline Analysis (DOA) methodology for facial analysis in dentistry.
This package uses [Haar
cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) for
face detection, a trained 68-facial-landmark model and statistical shape
analysis ([300 Faces In-The-Wild](https://ibug.doc.ic.ac.uk/resources/300-W/)).
The software uses an R script to conduct statistical
[shape](https://cran.r-project.org/web/packages/shapes/index.html) analysis
through a generalized Procrustes analysis (GPA), principal component analysis
(PCA) and non-parametric Goodall test, which compares mean shapes of each group
for significance. The script also computes mean Euclidean distance from a
baseline shape for each landmark.

This package was written to conduct automated facial analyses of patients
affected by Osteogenesis Imperfecta and controls under the BBDC 7701 study. Its
use may also be extended to the study of other dental and/or craniofacial
conditions or to compare different study groups while examining variables such
as sex, ethnicity, etc.

If you use this program or a modified version of it for research purposes
please cite as follows:

    @mybibtexref{

    :   title author year journal

    }

Features
--------

-   Takes 2 directories as input containing .jpg (anteroposterior
    clinical photographs)
-   Image Processing: scales images, transformation to grayscale
-   Detection: haar cascade face bounding, 68 facial landmark placement
-   Statistical Shape Analysis: GPA, PCA, Goodall's F-test, Euclidean
    distance per landmark from baseline shape

Requirements and Dependencies
-----------------------------

-   Python 3.5 (or higher)
-   Python packages: opencv-python, dlib, imutils, numpy, argparse, pandas,
    rpy2, progress
-   Linux operating system
-   R 3.3 (or more or higher)
-   R packages: shapes, foreach

Installation
------------

Important: in order for the required package rpy2 to install sucessfully, you
will need to have R version 3.3 or higher as well as the packages 'shapes' and
'foreach'

To install with **conda**:

```shell
conda env create -f environment.yml
conda activate pfla
mkdir shapes
cd shapes
conda skeleton cran --recursive shapes
conda build r-shapes
conda install -c local r-shapes
pip install pfla
```

To install with **pip**:

```shell
pip install -r requirements.txt
pip install pfla
```
Then in R
```R
install.packages("shapes", "foreach")
```

Usage
-----

To demonstrate the usage of the program we will be using images from the
Caltech Faces dataset which can be downloaded here
([male](https://github.com/maxrousseau/pfla/tree/master/pfla/test_males) and
[female](https://github.com/maxrousseau/pfla/tree/master/pfla/test_females)).

When using pfla, it is important to have your image directories structured in a
similar fashion.

The run the program, in the directory containing the image folders enter the
following:

```shell
$ pfla -g1 test_males  -g2 test_females
```

The resulting output from the analysis will be printed out into the
terminal like so:

```shell
*******************************
______________________________
___  __ \__  ____/__  /___    |
__  /_/ /_  /_   __  / __  /| |
_  ____/_  __/   _  /___  ___ |
/_/     /_/      /_____/_/  |_|
*******************************
Python Facial Landmark Analysis
Author: Maxime Rousseau
Source: https://github.com/maxrousseau/pfla

Processing Images |###############################| 10/10
g1 processing completed without errors
Processing Images |###############################| 10/10
g2 processing completed without errors

*Bootstrap - sampling with replacement within each group under H0: No of resamples =  10
******************************
null device
          1
[1] --------------------------------------------------------------------------------
[1] Goodall Statistical Test P-Value:  0.363636363636364
[1] --------------------------------------------------------------------------------
[1] Summary of Mean Euclidean Distance:
[1] Group 1:
[1] Mean:  0.00662911513379532 | Standard Deviation:  0.00257462207986629
[1] Group 2:
[1] Mean:  0.00743691647218815 | Standard Deviation:  0.00281889044033377
[1] --------------------------------------------------------------------------------
```

A histogram summarizing the mean Euclidean distances per landmark will
also be save in the data/ directory.

![Mean Euclidean Distance Histogram](paper/histo_02.png)

Testing
-------

To test your installation run the following commands:
```shell
cd ~/.local/lib/python3.5/site-packages/pfla/
python3 test.py
```
Documentation
-------------

Documentation of the package can be found here:
<https://pfla.readthedocs.io/en/latest/index.html#>

Contribute
----------

-   Refer to the contribution guidelines:
    <https://github.com/maxrousseau/pfla/blob/master/contributing.md>
-   Issue Tracker: <https://github.com/maxrousseau/pfla/issues>
-   Source Code: <https://github.com/maxrousseau/pfla>

License
-------

The project is licensed under the MIT license.

Contact
-------

Maxime Rousseau, DMD II McGill University, Faculty of Dentistry
- Email: <maximerousseau08@gmail.com>

