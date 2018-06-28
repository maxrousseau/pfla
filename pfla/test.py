# -*- coding: utf-8 -*-
import filecmp
import unittest
from fcn import img_prep
from fcn import face_detect
from fcn import annotate

class TestPfla(unittest.TestCase):
    """tests for the pfla package"""

    def setUp(self):
        """load image prepare and save in test directory"""
        self.test_raw = img_prep.RawImage("test/lena.jpg","img/img_prep/00_test.jpg","00_test")
        self.test_prep = self.test_raw.prepare()
        self.fdetector = face_detect.FaceDectector(
            "img/img_prep/00_test.jpg",
            "data/haarcascade_frontalface_default.xml",
            (100,100),
            (500,500),
            "00_test"
        )
        self.img_faces = self.fdetector.run_cascade()
        self.err = self.fdetector.to_matrix(self.img_faces)
        self.fannotator = annotate.FaceAnnotator(
            "00_test",
            "data/shape_predictor_68_face_landmarks.dat"
        )
        self.mat = self.fannotator.run_annotator()

    def test_img_prep(self):
        """test image preparation function"""
        success = filecmp.cmp("img/img_prep/00_test.jpg", "test/lena_gray.jpg", shallow=False)
        self.assertTrue(success)

    def test_face_processing(self):
        """test face processing functions"""
        success = filecmp.cmp("img/img_proc/00_test.jpg", "test/lena_processed.jpg", shallow=False)
        self.assertTrue(success)

unittest.main()
