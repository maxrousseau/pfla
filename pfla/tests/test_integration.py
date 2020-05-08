# -*- coding: utf-8 -*-
import os

import pytest
import numpy as np

from pfla.img_prep import ImgPrep
from pfla.face_detect import FaceDetect
from pfla.annotate import FaceAnnotate

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_DATA_TEST = os.path.join(CURRENT_PATH, "data")

@pytest.fixture
def prep_fxt():
    img_path = os.path.join(PATH_DATA_TEST, "m01.jpg")
    dir_path = PATH_DATA_TEST

    return ImgPrep(img_path), ImgPrep(dir_path)

def test_prep(prep_fxt):
    single_im, multiple_im = prep_fxt

    s_array, s_index =  single_im.prepare_file()
    assert len(s_index) == 1
    assert isinstance(s_array, np.ndarray)

    m_array, m_index =  multiple_im.prepare_dir()
    assert len(m_index) == 5
    assert isinstance(m_array, np.ndarray)

@pytest.fixture
def detect_fxt(prep_fxt):
    single_im, multiple_im = prep_fxt
    s_array, s_index =  single_im.prepare_file()
    m_array, m_index =  multiple_im.prepare_dir()

    return FaceDetect(s_array, True), FaceDetect(m_array, False)

def test_detect(detect_fxt):
    single_det, multiple_det = detect_fxt

    s_box = single_det.mtcnn_box()
    assert len(s_box) == 1

    m_box = multiple_det.mtcnn_box()
    assert len(m_box) == 5

@pytest.fixture
def annotate_fxt(prep_fxt, detect_fxt):
    single_im, multiple_im = prep_fxt
    s_array, s_index =  single_im.prepare_file()
    m_array, m_index =  multiple_im.prepare_dir()

    single_det, multiple_det = detect_fxt

    s_box = single_det.mtcnn_box()
    m_box = multiple_det.mtcnn_box()

    return FaceAnnotate(s_array, s_box, True), FaceAnnotate(m_array, m_box, False)


def test_annotate(annotate_fxt):
    single_ant, multiple_ant = annotate_fxt

    s_ldmk = single_ant.get_ldmk()
    assert isinstance(s_ldmk, np.ndarray)
    assert np.shape(s_ldmk) == (68, 2)
