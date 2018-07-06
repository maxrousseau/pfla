# -*- coding: utf-8 -*-
import filecmp
import unittest
from fcn import img_prep
from fcn import face_detect
from fcn import annotate
import sys
import os

for p in sys.path:
    if 'packages' in p:
        mod_path = p
mod_path = mod_path + '/pfla'

class TestPfla(unittest.TestCase):
    """tests for the pfla package"""

    def setUp(self):
        """load image prepare and save in test directory"""
        self.test_raw = img_prep.RawImage(
            mod_path + "/test/lena.jpg",
            mod_path + "/img/img_prep/00_test.jpg",
            "00_test")
        self.test_prep = self.test_raw.prepare()
        self.fdetector = face_detect.FaceDectector(
            mod_path + "/img/img_prep/00_test.jpg",
            mod_path + "/data/haarcascade_frontalface_default.xml",
            (100,100),
            (500,500),
            "00_test"
        )
        self.img_faces = self.fdetector.run_cascade()
        self.err = self.fdetector.to_matrix(self.img_faces)
        self.fannotator = annotate.FaceAnnotator(
            "00_test",
            mod_path + "/data/shape_predictor_68_face_landmarks.dat"
        )
        self.mat = self.fannotator.run_annotator()

    def test01_img_prep(self):
        """test image preparation function"""
        success = filecmp.cmp(
            mod_path + "/img/img_prep/00_test.jpg",
            mod_path + "/test/lena_gray.jpg",
            shallow=False)
        self.assertTrue(success)

    def test02_face_processing(self):
        """test face processing functions"""
        success = filecmp.cmp(
            mod_path + "/img/img_proc/00_test.jpg",
            mod_path + "/test/lena_processed.jpg",
            shallow=False)
        self.assertTrue(success)

unittest.main()
