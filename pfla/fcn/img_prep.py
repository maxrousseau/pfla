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
    GRAY : boolean
        Convert image to grayscale (default: False)

    Returns
    -------
    np_im : numpy array
        Numpy array of image(s)
    """
    def __init__(self, PATH, GRAY=False):
        self.path = PATH
        self.exts = ['jpg', 'png', 'tiff', 'bmp']
        self.gray = GRAY

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
        ls_path = []
        for ext in self.exts:
            dir_im = os.path.abspath(''.join([self.path, "/*.", ext]))
            ls_path.extend(glob.glob(dir_im))
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
