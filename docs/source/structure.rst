Structure
=========


The __init__.py file comprises of the main method calls while the different
classes are stored in the fcn/ directory. Under this directory, we find:
img_prep.py which will prepare the image by scaling and transforming it to
grayscale, face_detect.py which runs the haar cascade detecting the face on
the prepared image, annotate.py which places the landmarks on the detected
faces of the image, analyze.py calls the stats.R script which runs the
statistical analyses for the study.


The output images are stored as they are processed in their respective
directories: img_raw/ for the raw inputed images, img_prep/ for the prepared
images, img_proc/ for the processed images (faces detected and landmarks
placed).


The data/ directory contains the cascade classifier and shape predictor. Under
faces/ are stored the coordinates of the rectangles from the detected faces in
each image. The ldmks/ directory contains the matrices of the landmarks for
each groups to be analyzed using the R script.

The gross structure of the package is outlined below:

.. code-block:: shell

        pfla
        ├── contributing.md
        ├── docs
        │   ├── build
        │   ├── make.bat
        │   ├── Makefile
        │   └── source
        │       ├── analyze.rst
        │       ├── annotate.rst
        │       ├── conf.py
        │       ├── face_detect.rst
        │       ├── img_prep.rst
        │       ├── index.rst
        │       ├── install.rst
        │       ├── modules.rst
        │       ├── overview.rst
        │       └── structure.rst
        ├── LICENSE.txt
        ├── MANIFEST.in
        ├── paper
        │   ├── histo_02.png
        │   ├── paper.bib
        │   ├── paper.md
        │   └── pfla.png
        ├── pfla
        │   ├── annotate.py
        │   ├── cli.py
        │   ├── face_detect.py
        │   ├── img_prep.py
        │   ├── __init__.py
        │   ├── linear.py
        │   ├── logger.py
        │   ├── metrics.py
        │   └── tests
        │       ├── data
        │       │   ├── __init__.py
        │       │   ├── m01.jpg
        │       │   ├── m02.jpg
        │       │   ├── m03.jpg
        │       │   ├── m04.jpg
        │       │   └── m05.jpg
        │       ├── __init__.py
        ├── PROGRESS.md
        ├── README.md
        ├── requirements-pytorch.txt
        ├── requirements.txt
        └── setup.py

