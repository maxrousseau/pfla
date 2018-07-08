---
title: 'pfla: A Python Package for Dental Facial Analysis using Computer Vision
and Statistical Shape Analysis'
tags:
  - Python
  - R
  - facial analysis
  - dentistry
  - statistical shape analysis
  - euclidean geometry
authors:
  - name: Maxime Rousseau 
    orcid: 0000-0002-1417-2511 
    affiliation: 1
  - name: Jean-Marc Retrouvey
    orcid: 0000-0003-2112-9201
    affiliation: 1
affiliations:
 - name: McGill University, Faculty of Dentistry
   index: 1
 - name: Brittle Bone Disease Consortium
   index: 2
date: 17 May 2018
bibliography: paper.bib

---
# Summary
In this paper we will introduce a new methodology for clinical image analysis,
the Detection Outline Analysis (DOA) framework. This technique uses statistical
shape analysis and computer vision. To demonstrate its efficiency, we will
present a python package using this method for automatic analysis antero
posterior pictures of dental patients. The source code will be made available for the
benefits of the scientific community.

The goal of this paper is to outline the workings of the software used to conduct automatic
facial analyses on patients of the BBDC 7701 protocol on the natural history of Osteogenesis Imperfecta
and to ensure reproducibility of the study.

The program takes as input two folders of dental anteroposterior .jpg images. 
Each image it takes as input the object of interest is first detected, then landmarks are
assigned to this object. All coordinates for these landmarks are stored in
matrices for the different groups being compared. Finally through statistical
shape analysis, the groups are tested for the presence of statistical
differences in shapes or other analyses of interest to the researcher.

One of the main advantages of using this method for clinical image analysis is 
that the landmarks will be automatically
placed by the software, allowing us to standardise and greatly speed up the measurement procedures. 
The computer program  written in the
python and R[@r] programming languages using the OpenCV[@opencv] and Dlib[@dlib09] libraries as well as the publicly 
available facial annotation tool by [@sagonas13]
We will then store the 
coordinates of these landmarks in a two dimensional matrix using a csv type
file for later use in our statistical analysis.  


As the objective of this paper are both to describe a package applying specific
trained models and statistical analysis and presenting the broader outline of a
framework for facial analysis, it is important to understand that algorithms
can be interchanged(i.e. YOLO[@redmon16]) and the R script modified to suit the needs of a particular study.


The \_\_init\_\_.py file comprises of the main method calls while the different
classes are stored in the fcn/ directory. Under this directory, we find:
img\_prep.py which will prepare the image by scaling and transforming it to
grayscale, face\_detect.py which runs the haar cascade detecting the face on
the prepared image, annotate.py which places the landmarks on the detected
faces of the image, analyze.py calls the stats.R script which performs the
statistical analyses for the study.


The output images are stored as they are processed in their respective
directories: img\_raw/ for the raw inputed images, img\_prep/ for the prepared
images, img\_proc/ for the processed images (face bound by a rectangle and landmarks
placed).


The data/ directory contains the cascade classifier and shape predictor. Under
faces/ are stored the coordinates of the rectangles from the detected faces in
each image. The ldmks/ directory contains the matrices of the landmarks for
each groups to be analyzed using the R script.

When processing images, it first scales them to a set size to assure the
uniformity of the dataset. It is important to note that the images are not cropped hence
aspect ratios should be similar across the whole set of images. These are
then transformed to grayscale. 
After the initial preparation, the images then go through a
Haar Cascade classifier which was trained to detect faces\cite{viola01}. This
algorithm functions by scanning the input through the scope of a small
rectangle. It sums up the mean features of that said rectangle then compares it
to sections of the face training set. For our case, the algorithm was trained
on faces, hence it may recognize facial features such as eyes, noses, etc. 
This allows us to draw a bounding box around the face detected from the input
image.
Once the initial image processing is completed 
a landmark template is applied to the detected faces which produces a
matrix of 68 (x, y) coordinates for each patients. The outputed matrices are used to
compare groups of patients with clinical conditions to help us detect facial
manifestations of a disease. 

The antero posterior analaysis will consist of $l=68$ landmarks automatically 
placed on patient photographs by a computer program.
These sets of coordinates will produce matrices of $k=2$ dimensions.
The matrices will be represented as such: 
<center>
$M_{patient}=[x_1,x_2,...,x_l,y_1,y_2,...,y_l]$
</center>

Where $l$ represents the number of points attached to a photographs.


Statistical shape analysis has mostly been used in the field of evolutionary
biology for the analysis of skeletal artifact. It also has applications in the
medical fields, most notably in imaging analysis. Using the matrices produced
by the process explained above 
We will be using the matrices generated from the image 
processing in order to conduct the statistical analysis. The R script used the "shapes" package by (cite dryden).
First, we must
align the various matrices produced by the whole of our data. We will 
be doing so by doing a Generalized Procruste Analysis (GPA). This will allow 
us to work with shape matched in proportion and orientation. This is needed in 
our case given that we are interested in morphological differences. The
algorithm operates as follows:
1. arbitrarily choose a reference shape (usually from available instances)
2. superimpose all instances to current reference shape
3. compute mean shape of the curent set of superimposed shapes
4. if the Procruste distance between the mean shape and the reference shape is above a given threshold, set reference to mean shape and reiterate from step 2 



Once our 
matrices have been aligned we will transform them into unidimensional 
matrices through orthogonal projection by performing a Principal 
Component Analysis (PCA). This will aid us highlight the features present in
the dataset in order to facilitate comparison between groups. 
The vectors produced by this linearization of our
datasets will be annotated as such:
<center>
$V_{patient}=[i_1,i_2,...,i_{2l}]$
</center>

Following the PCA, we will be conducting a Goodall F test on the mean shapes of each
group using the non-parametric Bootstrap method to compare our multivariate 
matrices[@brombin09]. Two reasons explain our choice for this test, it is unreasonable 
to assume isotropy as well equal covariance between the population matrices 
being studied. This can be explained by the simple fact that our matrices
contain coordinates of human faces which are asymmetrical.
After having tested our hypothesis we will continue to explore the data in the
hopes of finding an explanation for the results of the hypothesis test. This
will be accomplished by computing the mean euclidean distance of each landmark
from its corresponding landmark on baseline shape. We then compile the given
values for each landmarks in a particular group resulting in a set of mean
distances per landmark from baseline. This will allow us interpret
the results on a deeper level as we will be able to isolate where are the
greatest differences and similarities between the study subjects. We can also
assess more broadly the mean and standard deviation of a particular set of
landmark.

It is important to understand that this is a morphological analysis, hence only
relative shape is evaluated. Conclusions related to size can not be drawn from
this method. 


We can visualize the accuracy of the image processing by inspecting the
detected face and landmarks  (Figure 1). The program also outputs a histogram
of the mean euclidean distance from baseline for each group (Figure 2). where
we have the female group in red and the male group in green.


Automatic facial landmarking and statistical shape analysis has proved to be a
reliable, reproducible and efficient way of conducting facial analysis. 

\newpage\

![Image Processing Example Over the Famous Lena Image](collage.png)

![Mean Euclidean Distance Output Histogram](histo_02.png)

\newpage

# References
