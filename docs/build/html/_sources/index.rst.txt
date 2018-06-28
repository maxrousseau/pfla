Python Facial Landmark Analysis
===============================

Introduction
------------

The Detection Outline Analysis (DOA) framework for image analysis methodology
was used to write the python package featured in this article. For each image
it takes as input the object of interest is first detected, then landmarks are
assigned to this object. All coordinates for these landmarks are stored in
matrices for the different groups being compared. Finally through statistical
shape analysis, the groups are tested for the presence of statistical
differences in shapes or other analyses of interest to the researcher.

One of the main advantages for clinical image analysis following this method is 
that the landmarks will be automatically
placed by the software, allowing us to standardise the measurement procedures. 
The computer program  written in the
python and R programming languages using the OpenCV and Dlib libraries as well as the publicly 
available facial annotation tool (300 Faces In-The-Wild)
We will then store the 
coordinates of these landmarks in a two dimensional matrix using a csv type
file for later use in our statistical analysis.  

As the objective of this paper are both to a package applying specific
trained models and statistical analysis and presenting the broader outline of a
framework for facial analysis, it is important to understand that algorithms
can be interchanged (i.e. YOLO).

Contents
--------

.. toctree::
   :maxdepth: 2

   install
   overview
   structure




Modules
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

