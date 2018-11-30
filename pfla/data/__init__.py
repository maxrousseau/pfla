import os
from ..fcn.fetcher import data_fetch

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def path_haar_cascade_front_face():
    """Return the path of the XML containing the Haar filters.
    """
    return os.path.join(CURRENT_PATH, 'haarcascade_frontalface_default.xml')


def path_shape_predictor():
    """Return the path to the data used to predict the landmark.

    The data will be downloaded the first time.
    """
    filename = os.path.join(CURRENT_PATH,
                            'shape_predictor_68_face_landmarks.dat')
    if not os.path.isfile(filename):
        data_fetch(filename)
    return filename
