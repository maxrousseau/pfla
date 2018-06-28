Overview
========


Image Processing
----------------

This program takes as inputs dental images in JPG (.jpg) format for initial
processing and prepare for landmarking and analyis. It first scales the image
to a set size to assure the uniformity of the dataset. They are then
transformed to grayscale. It is important to note
that the images are not cropped hence aspect ratios should be similar across the whole
dataset of images. 

Following the initial preparation, the images then go through a
Haar Cascade classifier which was trained to detect faces (algorithm included
in the opencv library). This
algorithm functions by scanning the input through the scope of a small
rectangle. It sums up the mean features of that said rectangle then compares it
to sections of the face training set. For our case, the algorithm was trained
on faces, hence it may recognize facial features such as eyes, noses, etc. 
This allows us to draw a bounding box around the face detected from the input
image.

Landmarks
---------

Once the initial image processing is completed 
a landmark template is applied to the detected faces which produces a
matrix of 68 (x, y) coordinates for each patients. 
These sets of coordinates will produce matrices of two dimensions [132 x (n patients)].

For each group of image being processed a separate csv file will be written
with all of the coordinates of the patients. 

Statistical Analysis
--------------------

After the coordinates of the landmarks have been written to a csv file. An R
script is called which runs statistical analyses comparing the two studied
groups. The results will be printed out to the terminal and the histogram for
this particular analysis saved under the data/ldmks/ folder of the package
directory.


