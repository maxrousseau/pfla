import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

from progress.bar import IncrementalBar

from .fcn import img_prep
from .fcn import face_detect
from .fcn import annotate
from .fcn import analyze
from .fcn import fetcher


def create_parser_pfla():
    parser = argparse.ArgumentParser(
        prog='pfla',
        description='Python facial landmark analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g1', '--group1', required=True,
                        help='Path to group 1 image directory')
    parser.add_argument('-g2', '--group2', required=True,
                        help='Path to group 2 image directory')
    return parser




def img_processing(img_id):
    """preparation, face detection and landmarking"""

    # begin by creating image object and preparing for processing
    img = img_prep.RawImage(
        os.path.join(mod_path, "img", "img_raw", "{}.jpg".format(img_id)),
        os.path.join(mod_path, "img", "img_prep", "{}.jpg".format(img_id)),
        img_id
    )
    img.prepare()

    # detect faces in image through Haar Cascade
    fdetector = face_detect.FaceDectector(
        os.path.join(mod_path, "img", "img_prep", "{}.jpg".format(img_id)),
        os.path.join(
            mod_path, "data", "haarcascade_frontalface_default.xml"),
        (100, 100),
        (500, 500),
        img_id,
        mod_path
    )
    img_faces = fdetector.run_cascade()
    err = fdetector.to_matrix(img_faces)

    # annotate the detected faces
    fannotator = annotate.FaceAnnotator(
        img_id,
        os.path.join(mod_path, "data",
                     "shape_predictor_68_face_landmarks.dat"),
        mod_path
    )
    mat = fannotator.run_annotator()

    return (mat, err)


def group_process(group, img_dir):
    """processing of a group of images"""
    input_dir = img_dir
    img_no = 0
    list_mat = []
    ib = IncrementalBar("Processing Images", max=len(glob.glob(img_dir)))
    errors = []

    for raw_img in sorted(glob.glob(input_dir)):

        # save images to be analyzed in the img_raw directory
        img_id = os.path.join(group, "{}".format(img_no))
        img = cv2.imread(raw_img)
        original_path = os.path.join(
            mod_path, "img", "img_raw", "{}.jpg".format(img_id))
        boolean = cv2.imwrite(original_path, img)

        ip_ret = img_processing(img_id)
        img_no += 1
        mat2 = ip_ret[0]
        if len(ip_ret[1]) > 1:
            errors.append(ip_ret[1])

        list_mat.append(list(mat2.values.flatten()))
        ib.next()

    all_mat = pd.DataFrame(list_mat)
    all_mat.to_csv(os.path.join(mod_path, "data", "ldmks",
                                "{}_landmark_matrix.csv".format(group)))
    if len(errors) != 0:
        print("\nWARNING: {} processing completed with errors:".format(group))
        for st in errors:
            print(st)
    else:
        print("\n {} processing completed without errors".format(group))
    ib.finish()


def pfla():
    mod_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(mod_path, 'art.txt'), 'r') as greet:
        shutil.copyfileobj(greet, sys.stdout)
    parser = create_parser_pfla()
    args = parser.parse_args()

    g1_img_dir = os.path.abspath(os.path.join(args["group1"], "*.jpg"))
    g2_img_dir = os.path.abspath(os.path.join(args["group2"], "*.jpg"))

    shape_pred = os.path.join(
        mod_path, "data", "shape_predictor_68_face_landmarks.dat")

    if not os.path.isfile(shape_pred):
        print('Shape predictor absent, downloading...')
        fetcher.data_fetch(shape_pred)


    for group in groups:
        # create directories for storage
        img_raw_d = (os.path.join(mod_path, "img", "img_raw"))
        img_prep_d = (os.path.join(mod_path, "img", "img_prep"))
        img_proc_d = (os.path.join(mod_path, "img", "img_proc"))
        data_d = (os.path.join(mod_path, "data", "faces"))
        list_dir = [img_raw_d, img_prep_d, img_proc_d, data_d]
        for direc in list_dir:
            if os.path.isdir(os.path.join(direc, group)):
                continue
            else:
                os.makedirs(os.path.join(direc, group))

        # clear directories before running functions
        img_raw_gd = (os.path.join(mod_path, "img", "img_raw", group, "*"))
        img_prep_gd = (os.path.join(mod_path, "img", "img_prep", group, "*"))
        img_proc_gd = (os.path.join(mod_path, "img", "img_proc", group, "*"))
        data_gd = (os.path.join(mod_path, "data", "faces", group, "*"))
        list_group_dir = [img_raw_gd, img_prep_gd, img_proc_gd, data_gd]
        for direc in list_group_dir:
            direc = glob.glob(direc)
            for f in direc:
                os.remove(f)


    group_process('g1', g1_img_dir)
    group_process('g2', g2_img_dir)
