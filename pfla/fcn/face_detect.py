# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#
# This class will take as input the prepared image (resized, grayscale) the 
# chosen cascade classifier as well as the minimum/maximum rectangles size and
# detect all faces on the picture and return them
#
#------------------------------------------------------------------------------
import cv2
import csv

class FaceDectector(object):

    def __init__(self, img_path, cascade, min_size, max_size, img_id):
        """Haar cascade face detection

        Description:
            create a Haar cascade object to draw rectangular bounding boxes
            around the inputed images

        Args:
            [img_path](str): path to prepared image
            [cascade](str): path to Harr cascade
            [min_size](int): minimum size of the detected face
            [max_size](int): maximum size of the detected face
            [img_id](str): identification number of the image being processed

        """
        self.img = cv2.imread(img_path)
        self.cascade = cascade
        self.min_size = min_size
        self.max_size = max_size
        self.img_id = img_id
        self.err = ""

    def run_cascade(self):
       """Create cascae and run on img

       Description:
           create the Haar cascade classifier [face_cascade] and detector [face_detector]
           run the detection algorithm on the parsed image [self.img]

        Return:
            [faces_detect](int): number of faces detected, success = 1, fail > 1, < 1

        """
       face_cascade = cv2.CascadeClassifier(self.cascade)
       faces_detect = face_cascade.detectMultiScale(
           self.img,
           scaleFactor = 1.1,
           minSize = (self.min_size),
           maxSize = (self.max_size),
           flags = cv2.CASCADE_SCALE_IMAGE
       )

       # let user know of multiple faces detected in one image
       if len(faces_detect) > 1:
           self.err = str("- " + str(len(faces_detect)) + " faces detected in " + self.img_id)
           return faces_detect
       else:
           return faces_detect

    def to_matrix(self, img_faces):
        """Save detected faces

        Description:
            transform the coordinates to a matrix and save as a .csv file
            corresponding to the image identification

        Args:
            [img_faces](int): array containing the integer coordingnates of the
                bound box drawn around the detected face

        Return:
            [self.err](str): string warning if there was more than one face
                detected in the image

        """

        with open("data/faces/" + self.img_id + ".csv", 'w', newline="") as csvfile:
            face_writer = csv.writer(csvfile)
            for (x, y, w, h) in img_faces:
                face_writer.writerow([x, y, w, h])

        return self.err



