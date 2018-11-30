# -*- coding: utf-8 -*-
import filecmp
import os
import shutil
import subprocess

from pfla.fcn import img_prep
from pfla.fcn import face_detect
from pfla.fcn import annotate

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_DATA_TEST = os.path.join(CURRENT_PATH, "data")


def setup_module():
    dirs = ["img_raw", "img_prep", "img_proc"]
    for dir_ in dirs:
        path = os.path.join(PATH_DATA_TEST, dir_)
        os.makedirs(path)


def teardown_module():
    dirs = ["img_raw", "img_prep", "img_proc", "matrix"]
    for dir_ in dirs:
        path = os.path.join(PATH_DATA_TEST, dir_)
        shutil.rmtree(path, ignore_errors=True)


def test_integration():
    test_raw = img_prep.RawImage(
        os.path.join(PATH_DATA_TEST, "testpic.jpg"),
        os.path.join(PATH_DATA_TEST, "img_prep", "00_test.jpg"),
        "00_test"
    )
    test_raw.prepare()
    fdetector = face_detect.FaceDectector(
        os.path.join(PATH_DATA_TEST, "img_prep", "00_test.jpg"),
        (100, 100), (500, 500), "00_test",
        os.path.join(PATH_DATA_TEST, "matrix")
    )
    img_faces = fdetector.run_cascade()
    fdetector.to_matrix(img_faces)
    fannotator = annotate.FaceAnnotator(
        "00_test",
        fdetector.matrix_path,
        os.path.dirname(test_raw.newpath),
        os.path.join(PATH_DATA_TEST, "img_proc")
    )
    fannotator.run_annotator()

    success = filecmp.cmp(
        os.path.join(PATH_DATA_TEST, "img_prep", "00_test.jpg"),
        os.path.join(PATH_DATA_TEST, "testpic_gray.jpg"),
        shallow=False
    )
    assert success

    success = filecmp.cmp(
        os.path.join(PATH_DATA_TEST, "img_proc", "00_test.jpg"),
        os.path.join(PATH_DATA_TEST, "testpic_processed.jpg"),
        shallow=False
    )
    assert success


def test_command_line():
    subprocess.check_output("pfla")
