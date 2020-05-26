Python Facial Landmark Analysis
===============================

Introduction
------------

A simple command line interface to automate facial analysis. ``pfla`` uses a
pre-trained neural networks to detect faces and annotate them with 68
landmarks. The program also compyte four commonly used facial metrics. The
output is saved to a file to allow for easy statistical analysis by the user.

The program takes a single image or a directory of images as input. Depending on the options
specified by the user it can perform face detection, annotation and compute facial metrics. The output is formatted
into a dataframe and saved to a file specified by the user.

PFLA support the following image formats: jpg, png, bmp, tiff. The output files formats
supported are: csv, h5, xlsx, pkl.

Contents
--------

.. toctree::
   :maxdepth: 2

   install
   overview
   structure
   modules

Modules
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

