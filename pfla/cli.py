import argparse
import glob
import os
import shutil
import sys

import cv2
import pandas as pd

from progress.bar import IncrementalBar


from .fcn import img_prep
from .fcn import face_detect
from .fcn import annotate
from .fcn import analyze

CURRENT_PATH = os.getcwd()
mod_path = os.path.dirname(os.path.abspath(__file__))


def create_parser_pfla():
    parser = argparse.ArgumentParser(
        prog='pfla',
        description='Python facial landmark analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_test_males = os.path.join(mod_path, 'data', 'test_males')
    parser.add_argument('-g1', '--group1', required=False,
                        help='Path to group 1 image directory',
                        default=default_test_males)
    default_test_females = os.path.join(mod_path, 'data', 'test_females')
    parser.add_argument('-g2', '--group2', required=False,
                        help='Path to group 2 image directory',
                        default=default_test_females)
    return parser


def img_processing(img_id, list_dir):
    """preparation, face detection and landmarking"""

    # begin by creating image object and preparing for processing
    img = img_prep.RawImage(
        os.path.join(list_dir[0], "{}.jpg".format(img_id)),
        os.path.join(list_dir[1], "{}.jpg".format(img_id)),
        img_id
    )
    img.prepare()

    # detect faces in image through Haar Cascade
    fdetector = face_detect.FaceDectector(
        os.path.join(list_dir[1], "{}.jpg".format(img_id)),
        (100, 100),
        (500, 500),
        img_id,
        list_dir[3]
    )
    img_faces = fdetector.run_cascade()
    err = fdetector.to_matrix(img_faces)

    # annotate the detected faces
    fannotator = annotate.FaceAnnotator(
        img_id,
        fdetector.matrix_path,
        list_dir[0],
        list_dir[2]
    )
    mat = fannotator.run_annotator()

    return (mat, err)


def group_process(group, img_dir, list_dir):
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
            list_dir[0], "{}.jpg".format(img_id)
        )
        cv2.imwrite(original_path, img)

        ip_ret = img_processing(img_id, list_dir)
        img_no += 1
        mat2 = ip_ret[0]
        if len(ip_ret[1]) > 1:
            errors.append(ip_ret[1])

        list_mat.append(list(mat2.values.flatten()))
        ib.next()

    all_mat = pd.DataFrame(list_mat)
    p = os.path.join(list_dir[3], "ldmks")
    if not os.path.exists(p):
        os.makedirs(p)
    all_mat.to_csv(os.path.join(p, "{}_landmark_matrix.csv".format(group)))
    if len(errors) != 0:
        print("\nWARNING: {} processing completed with errors:".format(group))
        for st in errors:
            print(st)
    else:
        print("\n {} processing completed without errors".format(group))
    ib.finish()


def pfla():
    with open(os.path.join(mod_path, 'art.txt'), 'r') as greet:
        shutil.copyfileobj(greet, sys.stdout)
    parser = create_parser_pfla()
    args = parser.parse_args()

    g1_img_dir = os.path.abspath(os.path.join(args.group1, "*.jpg"))
    g2_img_dir = os.path.abspath(os.path.join(args.group2, "*.jpg"))

    # clean the directories
    img_raw_d = os.path.join(CURRENT_PATH, "img", "img_raw")
    img_prep_d = os.path.join(CURRENT_PATH, "img", "img_prep")
    img_proc_d = os.path.join(CURRENT_PATH, "img", "img_proc")
    data_d = os.path.join(CURRENT_PATH, "data", "faces")
    list_dir = [img_raw_d, img_prep_d, img_proc_d, data_d]
    for dir_ in list_dir:
        shutil.rmtree(dir_, ignore_errors=True)

    groups = ["g1", "g2"]
    for group in groups:
        for dir_ in list_dir:
            d = os.path.join(dir_, group)
            os.makedirs(d)

    group_process('g1', g1_img_dir, list_dir)
    group_process('g2', g2_img_dir, list_dir)
    analyze.main_method(os.path.join(list_dir[3], 'ldmks'))
