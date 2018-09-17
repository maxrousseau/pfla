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
