# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# This class will take as input a raw image and prepare it for further
# processing by resizing it and converting it to grayscale.
#
# -----------------------------------------------------------------------------
import cv2
import imutils


class RawImage:
    """Raw images to be prepared for processing.

    Will read the raw image from the folder, scale it, turn it to grayscale,
    and save it to ``/img/img_prep/`` under its identification number.

    Parameters
    ----------
    path : string
        Path to the raw image.
    newpath : string
        Path where the prepared image will be saved.
    iden : string
        Identification number of the image being prepared for processing.
    """
    def __init__(self, path, newpath, iden):
        self.path = path
        self.newpath = newpath
        self.iden = iden

    def prepare(self):
        """Load, resize, convert to grayscale and save image.

        Read raw image [raw_img], resize it [resize_mg], transfor it to
        grayscale [gray_img] and write it to the new path in
        ``/img/img_prep/``.
        """
        raw_img = cv2.imread(self.path)
        resized_img = imutils.resize(raw_img, width=500)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.newpath, gray_img)
