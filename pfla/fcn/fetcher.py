# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#   This function will fetch the face predictor data file
#   and insert it in data/ directory
#
# -----------------------------------------------------------------------------
import requests


def data_fetch(filename):
    url = ('https://github.com/maxrousseau/pfla/blob/master/pfla/data/'
           'shape_predictor_68_face_landmarks.dat?raw=true')
    r = requests.get(url, allow_redirects=True)
    with open(filename, 'wb') as f:
        f.write(r.content)
