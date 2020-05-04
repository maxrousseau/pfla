import os

from facenet_pytorch import MTCNN

class FaceDetect:
    """Detect faces on images

    Parameters
    ----------
    IMG : numpy array
        Numpy array of prepared image(s).
    """

    def __init__(self, IMG):
        self.img = IMG
        self.err = 0

    def mtcnn_box(self):
        """Bounding boxes from the MTCNN face detector"""
        detector = MTCNN()
        face = detector.detect(self.img)
        box = face[0][0]
        bounding_box = [(box[0] , box[1], box[2], box[3])]

        return bounding_box
