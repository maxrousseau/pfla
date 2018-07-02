pfla: Python Facial Landmark Analysis
=====================================

[![PyPI
license](https://img.shields.io/pypi/l/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI version fury.io](https://badge.fury.io/py/pfla.svg)](https://pypi.org/project/pfla/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pfla.svg)](https://pypi.org/project/pfla/)
[![Read the Docs](https://img.shields.io/readthedocs/pip.svg)](https://pfla.readthedocs.io/en/latest/)

![example](paper/collage.png)

Advances in artificial intelligence have enhanced the usability of these
technologies in a clinical setting. This python package introduces the use of a
Detection Outline Analysis (DOA) methodology for facial analysis in
dentistry. This package uses [Haar cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) for face detection, a trained
68-facial-landmark model and statistical shape analysis ([300 Faces In-The-Wild](https://ibug.doc.ic.ac.uk/resources/300-W/)). The software
uses an R script to conduct statistical [shape](https://cran.r-project.org/web/packages/shape/index.html<Paste>) analysis through a
generalized Procrustes analysis (GPA), principal component analysis
(PCA) and non-parametric Goodall test, which compares mean shapes of
each group for significance. The script also computes mean Euclidean
distance from a baseline shape for each landmark.

This package was written to conduct automated facial analyses of patients
affected by Osteogenesis Imperfecta and controls under the BBDC 7701 study. Its
use may also be extended to the study of other dental and/or craniofacial
conditions or to compare different study groups while examining variables such
as sex, ethnicity, etc. 

If you use this program or a modified version of it for research purposes please cite as follows:

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

-   python 3.5
-   opencv
-   linux (or unix operating system)
-   R 3.3 (or more recent)
-   R packages: shapes, foreach

Installation
------------

```shell
$ pip install pfla
```


Additionnal steps, the 68 landmark dat file is too large for pip packaging.
You can download it [here](pfla/data/shape_predictor_68_face_landmarks.dat).


Place the downloaded dat file in the following directory:

```shell
$ ~/.local/lib/python3.5/site-packages/pfla/data/
```

Usage
-----

The program is run through the terminal as follows:

```shell
$ pfla -g1 /path/to/first/group -g2 /path/to/second/group
```

The resulting output from the analysis will be printed out into the
terminal like so:

```shell
[ INFO:0] Initialize OpenCL runtime...
Processing Images |###############################| 68/68
g1 processing completed without errors
Processing Images |###############################| 32/32
g2 processing completed without errors
*
Bootstrap - sampling with replacement within each group under H0:
No of resamples =

50

*
*
*

--------------------------------------------------------------------

Goodall Statistical Test P-Value:  0.0196078431372549

--------------------------------------------------------------------

Summary of Mean Euclidean Distance:

Group 1:

Mean:  0.0049944135958874 | Standard Deviation:  0.00156292696370281

Group 2:

Mean:  0.00590442732720643 | Standard Deviation:  0.0018474508985488

---------------------------------------------------------------------
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

Documentation of the package can be found here: <https://pfla.readthedocs.io/en/latest/> 

Contribute
----------

-   Issue Tracker: <https://github.com/maxrousseau/pfla/issues>
-   Source Code: <https://github.com/maxrousseau/pfla>

License
-------

The project is licensed under the MIT license.

Contact
-------

Maxime Rousseau, DMD II McGill University, Faculty of Dentistry
- Email: <maximerousseau08@gmail.com>

