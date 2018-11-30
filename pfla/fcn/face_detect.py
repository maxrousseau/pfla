# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# This class will take as input the prepared image (resized, grayscale) the
# chosen cascade classifier as well as the minimum/maximum rectangles size and
# detect all faces on the picture and return them
#
# -----------------------------------------------------------------------------
import os

import cv2
import pandas as pd

from ..data import path_haar_cascade_front_face


class FaceDectector:
    """Detect faces on images using Haar-like features.

    Parameters
    ----------
    img_path : string
        Path to prepared image.
    min_size : integer
        Minimum size of the detected face.
    max_size : integer
        Maximum size of the detected face.
    img_id : string
        Identification number of the image being processed.
    matrix_path : string
        The path to store the detected faces into a pandas dataframe.
    cascade : string
        Path to Harr cascade xml file. By default, we use the front Haar
        cascade from OpenCV.
    """

    def __init__(self, img_path, min_size, max_size, img_id, matrix_path,
                 cascade=path_haar_cascade_front_face()):
        self.img = cv2.imread(img_path)
        self.cascade = cascade
        self.min_size = min_size
        self.max_size = max_size
        self.img_id = img_id
        self.matrix_path = matrix_path
        self.err = ""

    def run_cascade(self):
        """Creates the cascade and run it on the object image.

        Creates the Haar cascade classifier [face_cascade] and detector
        [face_detector] run the detection algorithm on the parsed image
        [self.img].

        Returns
        -------
        faces_detect : list of integer
            Number of faces detected, success = 1, fail > 1, < 1. Errors will
            be recorded and reported at the end of the end of the analysis.

        """
        face_cascade = cv2.CascadeClassifier(self.cascade)
        faces_detect = face_cascade.detectMultiScale(
            self.img,
            scaleFactor=1.1,
            minSize=self.min_size,
            maxSize=self.max_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # let user know of multiple faces detected in one image
        if len(faces_detect) > 1:
            self.err = "- {} faces detected in {}".format(len(faces_detect),
                                                          self.img_id)
        return faces_detect

    def to_matrix(self, img_faces):
        """Save detected faces.

        Transform the coordinates to a matrix and save as a .csv file
        corresponding to the image identification.

        Parameters
        ----------
        img_faces : list of integer
            Array containing the integer coordinates of the bound box drawn
            around the detected face.

        Return
        ------
        error : string
            String warning if there was more than one face detected in the
            image.
        """
        if not os.path.exists(self.matrix_path):
            os.makedirs(self.matrix_path)
        filename = os.path.join(self.matrix_path, "{}.csv".format(self.img_id))
        pd.DataFrame(img_faces).to_csv(filename, header=None, index=False)

        return self.err
