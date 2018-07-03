# -*- coding: utf-8 -*-
import sys
from pfla.fcn import img_prep
from pfla.fcn import face_detect
from pfla.fcn import annotate
from pfla.fcn import analyze
from progress.bar import IncrementalBar
import os
import argparse
import cv2
import glob
import pandas as pd
import numpy as np
import csv

for p in sys.path:
    if 'packages' in p:
        mod_path = p
mod_path = mod_path + '/pfla'

ap = argparse.ArgumentParser()
ap.add_argument(
    "-g1",
    "--group1",
    required=True,
    help="path to group 1 image directory"
)
ap.add_argument(
    "-g2",
    "--group2",
    required=False,
    help="path to group 2 image directory"
)
args = vars(ap.parse_args())
g1_img_dir = args["group1"] + "*.jpg"
g2_img_dir = args["group2"] + "*.jpg"

def img_processing(img_id):
    """preparation, face detection and landamarking"""
    img_id = str(img_id)

    # begin by creating image object and preparing for processing
    img = img_prep.RawImage(
        mod_path + "/img/img_raw/" + img_id + ".jpg",
        mod_path + "/img/img_prep/" + img_id + ".jpg",
        img_id
    )
    img.prepare()

    # detect faces in image through Haar Cascade
    fdetector = face_detect.FaceDectector(
        mod_path + "/img/img_prep/" + img_id + ".jpg",
        mod_path + "/data/haarcascade_frontalface_default.xml",
        (100, 100),
        (500, 500),
        img_id
    )
    img_faces = fdetector.run_cascade()
    err = fdetector.to_matrix(img_faces)

    # annotate the detected faces
    fannotator = annotate.FaceAnnotator(
        img_id,
        mod_path + "/data/shape_predictor_68_face_landmarks.dat"
    )
    mat = fannotator.run_annotator()

    return (mat, err)

def group_process(group, img_dir):
    """processing of a group of images"""
    img_no = 0
    list_mat = []
    ib = IncrementalBar("Processing Images", max=len(glob.glob(img_dir)))
    errors = []

    # clear directories before running functions
    ir = (mod_path + "/img/img_raw/" + group + "/*")
    ig = (mod_path + "/img/img_prep/" + group + "/*")
    ip = (mod_path + "/img/img_proc/" + group + "/*")
    da = (mod_path + "/data/faces/" + group + "/*")
    list_dir = [ir, ig, ip, da]
    for direc in list_dir:
        direc = glob.glob(direc)
        for f in direc:
            os.remove(f)


    for raw_img in sorted(glob.glob(img_dir)):

        # save images to be analyzed in the img_raw directory 
        img_id = group + "/" + str(img_no)
        img = cv2.imread(raw_img)
        cv2.imwrite(mod_path + "/img/img_raw/" + str(img_id) + ".jpg", img)

        ip_ret = img_processing(img_id)
        img_no += 1
        mat2 = ip_ret[0]
        if len(ip_ret[1]) > 1:
            errors.append(ip_ret[1])

        list_mat.append(list(mat2.values.flatten()))
        ib.next()


    all_mat = pd.DataFrame(list_mat)
    all_mat.to_csv(mod_path + "/data/ldmks/" + group + "_landmark_matrix.csv")
    if len(errors) != 0:
        print("\nWARNING: " + str(group) + " processing completed with errors:")
        for st in errors:
            print(st)
    else:
        print("\n" + str(group) + " processing completed without errors")
    ib.finish()

def main():
    """main method of the program"""
    group_process('g1', g1_img_dir)
    group_process('g2', g2_img_dir)
    analyze.main()

main()
