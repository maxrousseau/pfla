# -*- coding: utf-8 -*-
import os

import pytest
import numpy as np

from pfla.img_prep import ImgPrep
from pfla.face_detect import FaceDetect
from pfla.annotate import FaceAnnotate
from pfla.metrics import Metrics
from pfla.linear import Linear
from pfla.logger import Logger

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
    single_annotate, multiple_annotate = annotate_fxt

    s_ldmk = single_annotate.get_ldmk()
    assert isinstance(s_ldmk, np.ndarray)
    assert np.shape(s_ldmk) == (68, 2)

    m_ldmk = multiple_annotate.get_ldmk()
    assert isinstance(m_ldmk, np.ndarray)
    assert np.shape(m_ldmk) == (5, 68, 2)

@pytest.fixture
def metrics_fxt(annotate_fxt):
    single_annotate, multiple_annotate = annotate_fxt

    s_ldmk = single_annotate.get_ldmk()
    m_ldmk = multiple_annotate.get_ldmk()

    return Metrics(s_ldmk, True), Metrics(m_ldmk, False),

def test_metrics(metrics_fxt):
    single_metrics, multiple_metrics = metrics_fxt

    s_metrics = single_metrics.compute_metrics()
    assert isinstance(s_metrics, np.ndarray)
    assert np.shape(s_metrics) == (4,)

    m_metrics = multiple_metrics.compute_metrics()
    assert isinstance(m_metrics, np.ndarray)
    assert np.shape(m_metrics) == (4, 5)

def test_linear():
    a = np.array([1,1])
    b = np.array([2,2])

    ln = Linear(a[0], a[1], b[0], b[1])
    dist = ln.euc_dist()

    assert dist == np.sqrt(2)

def test_logger():
    v_logging = Logger(True)
    nv_logging = Logger(False)

    assert v_logging.info('test message of level info', 0) == None
    assert nv_logging.info('test message of level warning', 1) == None
