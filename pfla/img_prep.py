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
        """Load image and return as numpy array"""
        im = Image.open(self.path)

        if self.gray:
            im = self.grayscale(im)
        else:
            None

        np_im = np.asarray(im)
        index = [os.path.splitext(os.path.basename(self.path))[0]]

        return np_im, index

    def prepare_dir(self):
        """Load images and return as numpy array"""
        ls_path = []
        ls_im = []
        index = []

        for ext in self.exts:
            dir_im = os.path.abspath(''.join([self.path, "/*.", ext]))
            ls_path.extend(glob.glob(dir_im))


        for i in ls_path:
            im = Image.open(i)
            if self.gray:
                im = self.grayscale(im)
            else:
                None
            im = np.asarray(im)
            ls_im.append(im)
            index.append(os.path.splitext(os.path.basename(i))[0])

        np_im = np.asarray(ls_im)

        return np_im, index
