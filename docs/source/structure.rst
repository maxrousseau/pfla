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
        ├── bin
        │   └── pfla
        ├── docs
        ├── LICENCE.txt
        ├── MANIFEST.in
        ├── paper
        ├── pfla
        │   ├── data
        │   │   ├── faces
        │   │   │   ├── g1
        │   │   │   └── g2
        │   │   ├── haarcascade_frontalface_default.xml
        │   │   ├── ldmks
        │   │   │   ├── g1_landmark_matrix.csv
        │   │   │   └── g2_landmark_matrix.csv
        │   │   └── shape_predictor_68_face_landmarks.dat
        │   ├── fcn
        │   │   ├── analyze.py
        │   │   ├── annotate.py
        │   │   ├── face_detect.py
        │   │   ├── img_prep.py
        │   │   ├── __init__.py
        │   │   └── stats.R
        │   ├── img
        │   │   ├── img_prep
        │   │   │   ├── g1
        │   │   │   └── g2
        │   │   ├── img_proc
        │   │   │   ├── g1
        │   │   │   └── g2
        │   │   └── img_raw
        │   │       ├── g1
        │   │       └── g2
        │   ├── __init__.py
        │   ├── test
        │   │   ├── lena_gray.jpg
        │   │   ├── lena.jpg
        │   │   └── lena_processed.jpg
        │   └── test.py
        ├── README.md
        └── setup.py
