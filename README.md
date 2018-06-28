pfla: Python Facial Landmark Analysis
=====================================

<a href='https://pfla.readthedocs.io/en/latest/?badge=latest'>
    <img src='//readthedocs.org/projects/pfla/badge/?version=latest' alt='Documentation Status' />
</a>

Advances in artificial intelligence have enhanced the usability of these
technologies in a clinical setting. This python package makes use of the
Detection Outline Analysis (DOA) methodology for facial analysis in
dentistry. This package uses Haar cascades for face detection, a trained
68-facial-landmark model and statistical shape analysis. The software
uses an R script to conduct statistical shape analysis through a
generalized Procrustes analysis (GPA), principal component analysis
(PCA) and non-parametric Goodall test, which compares mean shapes of
each group for significance. The script also computes mean Euclidean
distance from a baseline shape for each landmark.

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

Installation
------------

```shell
$ pip install pfla
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

References
----------

