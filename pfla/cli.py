import argparse
import glob
import os
import logging

from pfla.img_prep import ImgPrep
from pfla.face_detect import FaceDetect
from pfla.annotate import FaceAnnotate
from pfla.metrics import Metrics
from pfla.logger import Logger

import numpy as np
import pandas as pd


def create_parser_pfla():
    parser = argparse.ArgumentParser(
        prog='pfla',
        description="""PFLA: python facial landmark analysis.
        This program will read the image(s) given as input and can apply a
        face detection algorithm, landmark placement and computation of metrics. The
        results are returned as a text stream.
        """,
        epilog='AUTHOR: Maxime Rousseau LICENSE: MIT',
    )
    parser.add_argument("path",
                        help="path to the image or directory of images",
                        type=str)
    parser.add_argument("-d", "--detect",
                        help="detect faces and output bounding box",
                        required=False,
                        action="store_true")
    parser.add_argument("-l", "--landmark",
                        help="annotate detected faces and output coordinates",
                        required=False,
                        action="store_true")
    parser.add_argument("-m", "--metrics",
                        help="compute metrics and output results",
                        required=False,
                        action="store_true")
    parser.add_argument("-o", "--output",
                        help="specify output filename and format/filetype of the data",
                        required=False,
                        default="out.csv")
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        required=False,
                        action="store_true")

    return parser

def get_images(PATH):
    in_path = os.path.abspath(PATH)
    exists = os.path.exists(in_path)
    is_file = os.path.isfile(in_path)
    new_im = ImgPrep(in_path, GRAY=False)

    if exists:
        if is_file:
            np_img, index = new_im.prepare_file()
        else:
            np_img, index  = new_im.prepare_dir()
    else:
        info('provided path is invalid', 1)
        exit()

    return np_img, is_file, index

def get_box(IMG, IS_FILE):
    img = IMG
    is_file = IS_FILE

    detector = FaceDetect(img, is_file)
    # to use MTCNN the image must be RGB
    box = detector.mtcnn_box()

    return box

def get_landmarks(IMG, BOX, IS_FILE):
    img = IMG
    box = BOX
    is_file = IS_FILE

    annotator = FaceAnnotate(img, box, is_file)
    landmarks = annotator.get_ldmk()

    return landmarks

def get_metrics(LDMK, IS_FILE):
    ldmk = LDMK
    is_file = IS_FILE

    met = Metrics(ldmk, is_file)
    res = met.compute_metrics()

    return res

def to_df(BOX_I, LDMK_I, RESULTS_I, OUTPUT, INDEX):
    box = BOX_I
    ldmk = LDMK_I
    results = RESULTS_I
    out = OUTPUT
    index_arr = INDEX

    df_list = []
    ls_68 = list(range(1,69))
    ls_68x = [str(i)+'x' for i in ls_68]
    ls_68y = [str(i)+'y' for i in ls_68]
    col_box =  ['b1','b2','b3','b4']
    col_met = ['fi','lf/fh','mw/fw','mw/fh']

    if out['box']:
        fmt_box = pd.DataFrame(box, columns = col_box)
        df_list.append(fmt_box)

    if out['landmarks']:
        ldmk_x = np.reshape(ldmk[:,0], (1,68))
        ldmk_y = np.reshape(ldmk[:,1], (1,68))
        fmt_ldmkx = pd.DataFrame(ldmk_x, columns = ls_68x)
        fmt_ldmky = pd.DataFrame(ldmk_y, columns = ls_68y)
        df_list.append(fmt_ldmkx)
        df_list.append(fmt_ldmky)

    if out['metrics']:
        fmt_res = pd.DataFrame([results], columns = col_met)
        df_list.append(fmt_res)

    full_df = pd.concat(df_list, axis=1).set_index(pd.Index(index_arr))

    return full_df

def to_format(BOX, LDMK, RESULTS, OUTPUT, IS_FILE, INDEX):
    box = np.array(BOX)
    ldmk = LDMK
    results = RESULTS
    out = OUTPUT
    is_file = IS_FILE
    index_array = INDEX

    if is_file:
        fmt_out = to_df(box, ldmk, results, out, index_array)

    else:
        results = np.transpose(results)
        fmt_out_list = []

        for i in range(len(box)):
            fmt_out_i = to_df([box[i]], ldmk[i],
                                results[i], out, [index_array[i]])
            fmt_out_list.append(fmt_out_i)

        fmt_out = pd.concat(fmt_out_list)

    return fmt_out

def out_list(ARGS):
    args = ARGS
    out_dict = {'metrics':False, 'box':False, 'landmarks':False}

    if not args.detect and not args. landmark and not args.metrics:
        out_dict['metrics'] = True
        out_dict['box'] = True
        out_dict['landmarks'] = True
    else:
        if args.detect:
            out_dict['box'] = True
        if args.metrics:
            out_dict['metrics'] = True
        if args.landmark:
            out_dict['landmarks'] = True

    return out_dict

def op_list(ARGS):
    args = ARGS
    ops = 2

    if args.detect:
        ops = 0

    if args.landmark:
        ops = 1

    if args.metrics:
        ops = 2

    return ops

def df_parse(DF, OUT_NAME):
    df = DF
    ftype = os.path.splitext(OUT_NAME)[1]
    out_path = os.path.abspath(OUT_NAME)

    # csv excel hdf5 pickle
    if ftype == '.csv':
        df.to_csv(out_path)

    elif ftype == '.h5':
        df.to_hdf(out_path, 'df')

    elif ftype == '.pkl':
        df.to_csv(out_path)

    elif ftype == '.xlsx':
        df.to_csv(out_path)

    else:
        print("[ERROR] Filetype not supported")

def pfla():
    parser = create_parser_pfla()
    args = parser.parse_args()

    ops = op_list(args)
    out = out_list(args)

    log = Logger(args.verbose)
    log.info("pfla started, version 0.1.2", 0)


    # perform operations based on arguments
    if ops == 0:
        prep_img, is_file, index_arr = get_images(args.path)
        placeholder = list(range(len(index_arr)))
        box = get_box(prep_img, is_file)
        fmt_output = to_format(box, placeholder, placeholder, out, is_file, index_arr)

    elif ops == 1:
        prep_img, is_file, index_arr = get_images(args.path)
        placeholder = list(range(len(index_arr)))
        box = get_box(prep_img, is_file)
        landmarks = get_landmarks(prep_img, box, is_file)
        fmt_output = to_format(box, landmarks, placeholder, out, is_file, index_arr)

    elif ops == 2:
        prep_img, is_file, index_arr = get_images(args.path)
        box = get_box(prep_img, is_file)
        landmarks = get_landmarks(prep_img, box, is_file)
        results = get_metrics(landmarks, is_file)
        fmt_output = to_format(box, landmarks, results, out, is_file, index_arr)

    df_parse(fmt_output, args.output)
    log.info('data frame saved to file: %s' % args.output, 0)
