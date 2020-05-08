import os

import numpy as np
from facenet_pytorch import MTCNN

class FaceDetect:
    """Detect faces on images

    Parameters
    ----------
    IMG : numpy array
        Numpy array of prepared image(s).
    """

    def __init__(self, IMG, IS_FILE):
        self.img = IMG
        self.is_file = IS_FILE
        self.err = 0

    def mtcnn_box(self):
        """Bounding boxes from the MTCNN face detector"""
        detector = MTCNN()
        if self.is_file:
            np_im = np.array(self.img) # create new writeable numpy array
            face = detector.detect(np_im)
            box = face[0][0]
            bounding_box = [(box[0] , box[1], box[2], box[3])]
        else:
            bounding_box = []
            for i in self.img:
                np_im = np.array(i)
                face = detector.detect(np_im)
                box = face[0][0]
                bounding_box.append((box[0] , box[1], box[2], box[3]))

        return bounding_box
