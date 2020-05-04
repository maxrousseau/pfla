# Script used for the computations of the paper: Machine Learning in
# Contemporary Orthodontics Author: Maxime Rousseau
#
# Pre-treatment analysis of patient in the scope of orthodontics.  In this
# script we will use deep learning face detection, landmark placement and
# multidimensional data-mining to determine the norms previously established in
# orthodontic litterature.
# (Proffit, p.149-155)
#
# We present novel methods of pre-treatment facial analysis via automated
# facial recognition, landmark placement and analysis.

import sys
import os
import csv
import numpy as np
import torch
from facenet_pytorch import MTCNN
import numpy.lib.recfunctions as rfn
import face_alignment
import cv2
import pandas as pd
from linear import Linear as ln
import scipy.stats as stats
import matplotlib as mp

# load images and prepare for processing using opencv
root_path = os.getcwd()
img_path = os.path.join(root_path, 'db/img/')
img_labels = os.listdir(img_path)
img_list = [ os.path.join(img_path, i) for i in img_labels]
csv_path = os.path.join(root_path, 'db/db_tags.csv')
dataset = []

# logging
def sys_log(LABEL, STAGE, STATUS):
    label = LABEL 

    if STATUS == 0:
        status = 'SUCCESS'
    else:
        status = 'ERROR'

    if STAGE == 0:
        stage = 'processing: '
    elif STAGE == 1:
        stage = 'storage: '
    else:
        stage = 'unknown: '

    message = '[LOG] ' + stage + label + ' => ' + status
    print(message)

# metric calculations
def compute_metrics(LDMKS):
    fh = ln(LDMKS[27][0], LDMKS[27][1], LDMKS[8][0], LDMKS[8][1])
    fw = ln(LDMKS[1][0], LDMKS[1][1], LDMKS[15][0], LDMKS[15][1])
    lf = ln(LDMKS[33][0], LDMKS[33][1], LDMKS[8][0], LDMKS[8][1])
    mw = ln(LDMKS[4][0], LDMKS[4][1], LDMKS[12][0], LDMKS[12][1])

    facial_index = fh.euc_dist() / fw.euc_dist()
    lf_fh_index = lf.euc_dist() / fh.euc_dist()
    mw_fw_index = mw.euc_dist() / fw.euc_dist()
    mw_fh_index = mw.euc_dist() / fh.euc_dist()

    metrics = [facial_index, lf_fh_index, mw_fw_index, mw_fh_index]

    return metrics


# face dection using mtcnn, landmark placement using face_alignment
def image_processing(IMG_PATH):
    img_no = IMG_PATH
    img_no = img_no.replace(img_path,'')
    for s in '.JjPpGg':
        img_no = img_no.replace(s,'')

    img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    face = detector.detect(img)
    box = face[0][0]
    bounding_box = [(box[0], box[1], box[2], box[3])]

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False, device='cpu',
                                      face_detector='folder')
    ldmks = fa.get_landmarks(img, detected_faces=bounding_box)
    ldmks = ldmks[0]

    if ldmks.shape == (68,2):
        sys_log(IMG_PATH, 0, 0)
    else:
        sys_log(IMG_PATH, 0, 1)

    return img_no, ldmks

# the important stuff
def statistics(DATA):
    data = np.array(DATA)
    data = np.core.records.fromarrays(data.T, names='index, fi, lfh, mwfw, mwfh', formats='<i8, <f8, <f8, <f8, <f8')
    data = np.sort(data, order='index')

    # separate male female
    img_tags = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_tags.append(row)
    tags = np.asarray(img_tags)
    tags = np.delete(tags, 0,0)

    tags = np.core.records.fromarrays(tags.T, names='index, age, gender', formats='<i8, <f4, S1')
    data = rfn.join_by('index', data, tags)

    female = data[0:3]
    male = data[4:]

    # (mean, sd) for each metric including age
    female_age = (np.mean(female['age']), np.std(female['age']))
    female_fi = (np.mean(female['fi']), np.std(female['fi']))
    female_lfh = (np.mean(female['lfh']), np.std(female['lfh']))
    female_mwfh = (np.mean(female['mwfh']), np.std(female['mwfh']))
    female_mwfw = (np.mean(female['mwfw']), np.std(female['mwfw']))

    male_age = (np.mean(male['age']), np.std(male['age']))
    male_fi = (np.mean(male['fi']), np.std(male['fi']))
    male_lfh = (np.mean(male['lfh']), np.std(male['lfh']))
    male_mwfh = (np.mean(male['mwfh']), np.std(male['mwfh']))
    male_mwfw = (np.mean(male['mwfw']), np.std(male['mwfw']))

    # data from 1987 study
    male_1987_fi = np.random.normal(0.885, 0.051, 50)
    male_1987_lfh = np.random.normal(0.592, 0.027, 50)
    male_1987_mwfh = np.random.normal(0.803, 0.068, 50)
    male_1987_mwfw = np.random.normal(0.708, 0.038, 50)

    female_1987_fi = np.random.normal(0.862, 0.046, 50)
    female_1987_lfh = np.random.normal(0.586, 0.029, 50)
    female_1987_mwfh = np.random.normal(0.817, 0.060, 50)
    female_1987_mwfw = np.random.normal(0.701, 0.042, 50)

    # ANOVA vs data from 1989
    female_fi_AOV = stats.f_oneway(female_1987_fi, female['fi'])
    female_lfh_AOV = stats.f_oneway(female_1987_lfh, female['lfh'])
    female_mwfh_AOV = stats.f_oneway(female_1987_mwfh, female['mwfh'])
    female_mwfw_AOV = stats.f_oneway(female_1987_mwfw, female['mwfw'])

    male_fi_AOV = stats.f_oneway(male_1987_fi, male['fi'])
    male_lfh_AOV = stats.f_oneway(male_1987_lfh, male['lfh'])
    male_mwfh_AOV = stats.f_oneway(male_1987_mwfh, male['mwfh'])
    male_mwfw_AOV = stats.f_oneway(male_1987_mwfw, male['mwfw'])

    male_stats = {'age': male_age, 'fi':male_fi,'lfh':male_lfh, 'mwfh':male_mwfh, 'mwfw':male_mwfw}
    female_stats = {'age': female_age, 'fi':female_fi,'lfh':female_lfh, 'mwfh':female_mwfh, 'mwfw':female_mwfw}

    male_pvalues = {'fi':male_fi_AOV[1],'lfh':male_lfh_AOV[1], 'mwfh':male_mwfh_AOV[1], 'mwfw':male_mwfw_AOV[1]}
    female_pvalues = {'fi':female_fi_AOV[1],'lfh':female_lfh_AOV[1], 'mwfh':female_mwfh_AOV[1], 'mwfw':female_mwfw_AOV[1]}

    return male_stats, male_pvalues, female_stats, female_pvalues

def main():
    # iteration through images
    for img in img_list:
        img_no, ldmk_data = image_processing(img)
        metrics = compute_metrics(ldmk_data)
        subject_data = [int(img_no)] + metrics
        dataset.append(subject_data)

    ms, mp, fs, fp = statistics(dataset)
    f = open('results.txt', 'w')
    f.write(str(ms) + '\n'  + str(fs) + '\n' + str(mp) + '\n' + str(fp))
    f.close()

if __name__ == "__main__":
    main()
