# -*- coding: utf-8 -*-
import glob
import os

import numpy as np

from PIL import Image
from PIL import ImageOps

class ImgPrep:
    """Raw images to be prepared for processing.

    Will read the raw image from the folder, scale it, turn it to grayscale,
    and save it to ``/img/img_prep/`` under its identification number.

    Parameters
    ----------
    PATH : string
        Path to the image or image directory.
    EXT : string
        Extension to the image(s).
    GRAY : boolean
        Convert image to grayscale (default: False)

    Returns
    -------
    np_im : numpy array
        Numpy array of image(s)
    """
    def __init__(self, PATH, EXT=None, GRAY=False):
        self.path = PATH
        self.ext = EXT
        self.gray = GRAY
        self.resize = RESIZE

    def grayscale(self, image):
        gray_im = ImageOps.grayscale(image)

        return gray_im

    def prepare_file(self):
        """Load, resize, convert to grayscale and save image."""
        im = Image.open(self.path)
        if self.gray:
            im = self.grayscale(im)
        else:
            None

        np_im = np.asarray(im)

        return np_im

    def prepare_dir(self):
        """Load, resize, convert to grayscale and save image."""
        dir_im = os.path.abspath(''.join([self.path, "*.", self.ext]))
        ls_path = glob.glob(dir_im)
        ls_im = []

        for i in ls_path:
            im = Image.open(i)
            if self.gray:
                im = self.grayscale(im)
            else:
                None
            im = np.asarray(im)
            ls_im.append(im)

        np_im = np.asarray(ls_im)

        return np_im
