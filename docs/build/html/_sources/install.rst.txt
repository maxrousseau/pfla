Installation and Usage
======================

Requirements and Dependencies
-----------------------------

-   Python 3.5
-   Python packages: opencv-python, dlib, imutils, numpy, argparse, pandas, rpy2, progress
-   Linux operating system
-   R 3.3 (or more recent)
-   R packages: shapes, foreach

Installation
------------

Important: in order for the required package rpy2 to install sucessfully, you
will need to have R version 3.3 or higher as well as the packages 'shapes' and
'foreach'

To install enter the following commands:

        $ pip install pfla


Additionnal steps, the 68 landmark dat file is too large for pip packaging.
You can download it [here](pfla/data/shape_predictor_68_face_landmarks.dat).

Place the downloaded dat file in the following directory:

        $ ~/.local/lib/python3.5/site-packages/pfla/data/
