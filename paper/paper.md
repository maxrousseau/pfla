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
# Introduction

This paper outlines the workings of the software used to conduct automatic
facial analyses on patients of the BBDC 7701 protocol investigating the natural
history of Osteogenesis Imperfecta (OI).

The software takes two image directories containing facial pictures of the study
subjects as inputs. Each image is processed through face detection and landmark
allocation producing a matrix of coordinates for each study group. Theses
matrices are then used for the statistical comparison of the groups. The
results from the analysis is printed to the terminal (Figure 1). 

![Schematic representation of the pfla software for OI facial
analysis](./pfla.png)

One of the main advantages of using this software for clinical image analysis
is that it automatically places landmarks, standardizing and expediting
measurement procedures. This task was previously conducted manually which was
very time consuming and prone to operator error and bias. With this novel
approach we are able to circumvent these issues allowing us to have an efficient and objective way of conducting facial analysis.


This paper simultaneously presents a broad framework for facial analysis, while
describing a specific package applying trained models and statistical analysis.
These algorithms can be interchanged (i.e. YOLO [@redmon16]) and the R
script modified to suit the needs of a particular study.



# Software

The program was  written in python and R [@r]
programming languages using OpenCV [@opencv] and Dlib [@dlib09]
libraries, as well as the publicly available facial annotation tool by
[@sagonas13].

The program takes as input two folders of dental anteroposterior .jpg images,
before assigning landmarks to each object of interest. All coordinates are
stored in separate matrices for each group being compared. They are then tested
using statistical shape analysis for differences in shape and other attributes
of interest to the researcher.

The ```__ init__.py``` file comprises the main method calls, while the classes
are stored in the ```fcn/``` directory. Under this directory can be found:
```img_prep.py```, which prepares images by rescaling and converting to grayscale;
```face_detect.py```, which runs the Haar cascade detecting the face on the prepared
image; ```annotate.py```, which places landmarks on the detected faces; and
```analyze.py```, which calls the ```stats.R``` script to perform statistical analyses.

The program stores output images in their respective directories: ```img_raw/``` for
raw inputted images, ```img_prep/``` for prepared images, and ```img_proc/``` for processed
images (face bound by a rectangle with landmarks placed).

The ```data/``` directory
contains the cascade classifier and shape predictor. Under ```faces/``` are stored
coordinates of the rectangles from the detected faces in each image. The ```ldmks/```
directory contains the matrices of landmarks for each group to be analyzed
using the R script.

## Image Processing

After initial preparation, images go through a Haar Cascade classifier trained
to detect faces [@viola01]. This algorithm scans the input through
the scope of a small rectangle. It sums up the mean features thus detected,
comparing them to sections of the face training set. For our purpose, the
algorithm was trained on faces, hence it may recognize facial features such as
eyes, noses, etc. This allows us to draw a bounding box around the face
detected in the input image. Once initial image processing is completed, a
landmark template is applied which produces a matrix of 68 (x, y) coordinates
for each patient. The outputted matrices help detect facial manifestations
of disease by comparing groups of patients with clinical conditions to controls.


The antero posterior analysis consists of $l=68$ landmarks automatically placed
on patient photographs via software. These sets of coordinates produce matrices
of $k=2$ dimensions. The matrix for a single study group are represented as such:

\begin{center}
$m^{[s]}=[x_1,x_2,...,x_{68},y_1,y_2,...,y_{68}]$
\end{center}

Where the $s$ superscript represents the subject attached to this row of
coordinates. 

## Statistical Shape Analysis

Statistical shape analysis has mostly been used in the field of evolutionary
biology for the analysis of skeletal artifacts [@dryden-shapes]. It also has applications in the
medical fields, most notably in imaging analysis. The pfla package performs the
image processing described above in order to conduct statistical analysis [@dryden92]. The
R script uses the "shapes" package by [@dryden16]. First, the various
matrices produced by our data are aligned. This is done by performing a
Generalized Procruste Analysis (GPA)[@gower75],[@ten77]. This allows for shapes  matched in
proportion and orientation. This is needed for the purpose of the study, given
the authors' interest in morphological differences. The algorithm operates
as follows:

1. arbitrarily choose a reference shape (usually from available instances) 
2. superimpose all instances to current reference shape 
3. compute mean shape of the current set of superimposed shapes 
4. if the Procruste distance between the mean shape and the reference shape is above a given threshold, set reference to mean shape and reiterate from step 2 

Once our matrices are aligned, they are
transformed into unidimensional matrices through orthogonal projection by
performing a Principal Component Analysis (PCA). This highlights features
present in the dataset in order to facilitate comparison between groups. The
vectors produced by the linearization of our datasets will be annotated as
such:

\begin{center}
$v^{[s]}=[i_1,i_2,...,i_{136}]$
\end{center}

Following the PCA, the Goodall F test [@goodall91] is computed on the mean shapes of each
group using the non-parametric Bootstrap method to compare multivariate
matrices [@brombin09]. It is unreasonable to assume isotropy as
well as equal covariance between the matrices being studied. This can be
explained by the simple fact that human faces are naturally asymmetrical. After
testing our hypothesis, exploration of the data is continued in the hope of
finding an explanation for these results. This is accomplished by computing the
mean Euclidean distance of each landmark from its corresponding landmark on the
baseline shape.  Given values for each landmark in a particular group are then
compiled, resulting in a set of mean distances per landmark from baseline. This
allows for interpretion the results on a deeper level, isolating the greatest
differences and similarities between study subjects. Mean and standard
deviation of a particular set of landmarks can therefore be assessed more
broadly.

It is important to understand that this is a morphological analysis,
hence only relative shape is evaluated. Conclusions related to size cannot be
drawn from this method.

Accuracy of the image processing can be visualized by
inspecting detected faces and landmarks. The program outputs a
histogram of mean Euclidean distances from the baseline for each group (Figure
2). 


![Mean Euclidean Distance Output Histogram](histo_02.png)

\newpage

# References
