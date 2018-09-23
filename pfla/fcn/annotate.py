# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#
# This class will take in the detected faces from the images and anotate them
# with 68 landmarks and output the resulting image and its coordinate matrix
#
#------------------------------------------------------------------------------
import os
import sys
import cv2
import dlib
import numpy as np
import csv
import pandas as pd

class FaceAnnotator(object):

    def __init__(self, img_id, classifier, mod_path):
        """
        Fit the 68 landmarks to the detected face.

        Initialization of the FaceAnnotator object.

        Parameters
        ----------
        img_id : string
            Identification number of the image being processed
        classifier : string
            Path to the classifier xml file containing 68 landmark algorithm
        mod_path : string
            Path to the pfla module

        Returns
        -------
        None
        """
        self.img_id = img_id
        self.classifier = classifier
        self.mod_path = mod_path

    def run_annotator(self):
        """
        Create predictor and run landmark detection algorithm.

        Fetch detected face bounding rectangle coordinates from
        corresponding csv file in data/face/, create dlib rectangle, use
        dlib shape predictor to place landmark from classifier, save matrix
        to pandas dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        dataframe : pandas
            matrix containing the coordinates of the 68 landmarks placed on the
            detected face.
        """
        rect = []
        with open(
            os.path.join(self.mod_path, "data/faces/", str(self.img_id + ".csv")),
            newline=""
            ) as csvfile:
            face_reader = csv.reader(csvfile)
            for row in face_reader:
                rect.append(row)

        # transform to dlib rectangle and create predictor  
        rect = rect[0]
        x1 = int(rect[0])
        x2 = int(rect[0]) + int(rect[2])
        y1 = int(rect[1])
        y2 = int(rect[1]) + int(rect[3])
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        predictor = dlib.shape_predictor(self.classifier)

        # place landmarks and convert to pandas dataframe
        img_prep = cv2.imread(os.path.join(self.mod_path, "img/img_prep/",
                                           str(self.img_id + ".jpg")))
        landmarks = predictor(img_prep, dlib_rect).parts()
        landmarks_np = np.matrix([[p.x, p.y] for p in landmarks])
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
                color=(0,0,255)
            )
            cv2.circle(img_proc, position, 1, color=(255, 0, 0))
        cv2.rectangle(img_proc, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imwrite(os.path.join(self.mod_path, "img/img_proc/", str(self.img_id + ".jpg")), img_proc)


        return dataframe







