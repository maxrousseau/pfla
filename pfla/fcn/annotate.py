# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# This class will take in the detected faces from the images and annotate them
# with 68 landmarks and output the resulting image and its coordinate matrix
#
# -----------------------------------------------------------------------------
import os
import sys

import cv2
import dlib
import numpy as np
import pandas as pd


class FaceAnnotator:
    """Automatic face landmark annotation.

    Parameters
    ----------
    img_id : string
        Identification number of the image being processed.
    classifier : string
        Path to the classifier xml file containing 68 landmark algorithm.
    mod_path : string
        Path to the pfla module.
    """

    def __init__(self, img_id, classifier, mod_path):
        self.img_id = img_id
        self.classifier = classifier
        self.mod_path = mod_path

    def run_annotator(self):
        """Create predictor and run landmark detection algorithm.

        Fetch detected face bounding rectangle coordinates from corresponding
        csv file in ``data/face/``, create ``dlib`` rectangle, use ``dlib``
        shape predictor to place landmark from classifier, save matrix to
        ``pandas`` dataframe.

        Returns
        -------
        dataframe : pandas dataframe
            Dataframe containing the coordinates of the 68 landmarks placed on
            the detected face.
        """
        filename = os.path.join(
            self.mod_path, "data", "faces", "{}.csv".format(self.img_id)
        )
        rect = np.genfromtxt(filename, delimiter=',', dtype=int)

        # transform to dlib rectangle and create predictor
        x1 = rect[0]
        x2 = rect[0] + rect[2]
        y1 = rect[1]
        y2 = rect[1] + rect[3]
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        predictor = dlib.shape_predictor(self.classifier)

        # place landmarks and convert to pandas dataframe
        img_prep = cv2.imread(
            os.path.join(self.mod_path, "img", "img_prep",
                         "{}.jpg".format(self.img_id))
        )
        landmarks = predictor(img_prep, dlib_rect).parts()
        landmarks_np = np.array([[p.x, p.y] for p in landmarks])
        dataframe = pd.DataFrame(landmarks_np)

        # draw and save annotated image
        img_proc = img_prep
        for idx, point in enumerate(landmarks_np):
            position = (point[0, 0], point[0, 1])
            cv2.putText(
                img_proc,
                str(idx),
                position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255)
            )
            cv2.circle(img_proc, position, 1, color=(255, 0, 0))
        cv2.rectangle(img_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(
            os.path.join(self.mod_path, "img", "img_proc",
                         "{}.jpg".format(self.img_id)),
            img_proc
        )

        return dataframe
