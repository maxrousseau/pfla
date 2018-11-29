# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# This class will take as input the prepared image (resized, grayscale) the
# chosen cascade classifier as well as the minimum/maximum rectangles size and
# detect all faces on the picture and return them
#
# -----------------------------------------------------------------------------
import os
import sys

import cv2
import pandas as pd


class FaceDectector:
    """Detect faces on images using Haar-like features.

    Parameters
    ----------
    img_path : string
        Path to prepared image.
    cascade : string
        Path to Harr cascade xml file.
    min_size : integer
        Minimum size of the detected face.
    max_size : integer
        Maximum size of the detected face.
    img_id : string
        Identification number of the image being processed.
    mod_path : string
        Path to the pfla module.
    """

    def __init__(self, img_path, cascade, min_size, max_size, img_id,
                 mod_path):
        self.mod_path = mod_path
        self.img = cv2.imread(img_path)
        self.cascade = cascade
        self.min_size = min_size
        self.max_size = max_size
        self.img_id = img_id
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
        filename = os.path.join(self.mod_path, "data", "faces",
                                "{}.csv".format(self.img_id))
        pd.DataFrame(img_faces).to_csv(filename, header=None, index=False)

        return self.err
