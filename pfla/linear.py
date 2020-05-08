# -*- coding: utf-8 -*-
import numpy as np

class Linear(object):
    """This class is a linear mathematical function"""

    def __init__(self, coords_ax, coords_ay, coords_bx, coords_by):
        """Linear mathematical function object
        Initialize with two coordinates to determine the constant and slope.

        Contains method to determine Y given X.

        Parameters
        ----------
        coords_ax : numpy array
            x coordinate for first landmark
        coords_ay : numpy array
            y coordinate for first landmark
        coords_bx : numpy array
            x coordinate for second landmark
        coords_bx : numpy array
            y coordinate for second landmark

        Returns
        ------
        None
        """

        self.x1 = coords_ax
        self.y1 = coords_ay
        self.x2 = coords_bx
        self.y2 = coords_by

    def euc_dist(self):
        """Compute the Euclidean distance between 2 landmarks

        Calculates the Euclidean distance between 2 landmarks and returns it as
        output.

        Parameters
        ----------
        NONE

        Returns
        -------
        distance : numpy array
            Euclidean between the two landmarks
        """
        height = self.y1 - self.y2
        width = self.x1 - self.x2
        distance = np.sqrt((np.square(height)+np.square(width)))

        return distance
