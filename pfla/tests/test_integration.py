# -*- coding: utf-8 -*-
import os

import pytest
import numpy as np

from pfla.img_prep import ImgPrep

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_DATA_TEST = os.path.join(CURRENT_PATH, "data")

@pytest.fixture
def image_preparation():
    img_path = os.path.join(PATH_DATA_TEST, "m01.jpg")
    dir_path = PATH_DATA_TEST

    return ImgPrep(img_path), ImgPrep(dir_path)

def test_image_preparation(image_preparation):
    single_im, multiple_im = image_preparation

    s_array, s_index =  single_im.prepare_file()
    assert len(s_index) == 1
    assert isinstance(s_array, np.ndarray)

    m_array, m_index =  multiple_im.prepare_dir()
    assert len(m_index) == 5
    assert isinstance(m_array, np.ndarray)
