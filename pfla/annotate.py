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

    def __init__(self, IMG, BOX, IS_FILE):
        self.img = IMG
        self.box = BOX
        self.is_file = IS_FILE

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

        if self.is_file:
            ldmk = fa.get_landmarks(self.img, self.box)
            ldmk = ldmk[0]
        else:
            ldmk = []
            for i in range(len(self.img)):
                img_i = np.array(self.img[i])
                box_i = [self.box[i]]
                ldmk_i = fa.get_landmarks(img_i, box_i)
                ldmk_i = ldmk_i[0]
                ldmk.append(ldmk_i)
            ldmk = np.asarray(ldmk)

        return ldmk
