Installation and Usage
======================

Requirements and Dependencies
-----------------------------

-   Python 3.5 (or higher)
-   Python packages: opencv-python, dlib, imutils, numpy, argparse, pandas, rpy2, progress
-   Linux operating system
-   R 3.3 (or higher)
-   R packages: shapes, foreach

Installation
------------

Important: in order for the required package rpy2 to install sucessfully, you
will need to have R version 3.3 or higher as well as the packages 'shapes' and
'foreach'

To install with **conda**:

.. code-block:: shell
  
  conda env create -f environment.yml
  source activate pfla
  mkdir shapes
  cd shapes
  conda skeleton cran --recursive shapes
  conda build r-shapes
  conda install -c local r-shapes
  pip install pfla

To install with pip:

.. code-block:: shell

  pip install -r requirements.txt
  pip install pfla
  
Usage
-----

To demonstrate the usage of the program we will be using images from the
Caltech Faces dataset which can be downloaded here
([male](https://github.com/maxrousseau/pfla/tree/master/pfla/test_males) and [female](https://github.com/maxrousseau/pfla/tree/master/pfla/test_females)). 

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

![Mean Euclidean Distance Histogram](../../paper/histo_02.png)
