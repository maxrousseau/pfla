# -*- coding: utf-8 -*-
import os

import face_alignment
import numpy as np


class FaceAnnotate:
    """Face annotation with 68 landmarks

    Parameters
    ----------
    IMG : numpy array
        Numpy array of the image to be processed for landmarks
    BOX : typle
        Bounding box of the detected face
    """

    def __init__(self, IMG, BOX):
        self.img = IMG
        self.box = BOX

    def get_ldmk(self):
        """Get landmark coordinates for detected face

        Returns
        -------
        ldmk : numpy array
            Numpy array containing the coordinates of the landmarks
        """
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                        flip_input=False,
                                        device='cpu',
                                        face_detector='folder')
        ldmk = fa.get_landmarks(self.img, self.box)
        ldmk = ldmk[0]

        return ldmk
