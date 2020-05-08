#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from pfla.linear import Linear as ln

class Metrics:
    """Compute various metrics based on landmarks

    Parameters
    ----------
    LDMK: numpy array
        Array containing the landmarks for the detected face

    IS_FILE: boolean
        Is the input a file

    Returns
    -------
    metrics: numpy array
        Numpy array of the metrics computed
    """

    def __init__(self, LDMK, IS_FILE):
        self.ldmk = LDMK
        self.is_file = IS_FILE

    def compute_ratio(self, coord):
        fh = ln(coord[27][0], coord[27][1],
                coord[8][0],  coord[8][1])
        fw = ln(coord[1][0],  coord[1][1],
                coord[15][0], coord[15][1])
        lf = ln(coord[33][0], coord[33][1],
                coord[8][0],  coord[8][1])
        mw = ln(coord[4][0],  coord[4][1],
                coord[12][0], coord[12][1])

        return fh, fw, lf, mw

    def compute_metrics(self):
        if self.is_file:
            fh, fw, lf, mw = self.compute_ratio(self.ldmk)

            facial_index = fh.euc_dist() / fw.euc_dist()
            lf_fh_index = lf.euc_dist() / fh.euc_dist()
            mw_fw_index = mw.euc_dist() / fw.euc_dist()
            mw_fh_index = mw.euc_dist() / fh.euc_dist()

        else:
            facial_index = []
            lf_fh_index = []
            mw_fw_index = []
            mw_fh_index = []

            for i in self.ldmk:
                fh, fw, lf, mw = self.compute_ratio(i)

                facial_index.append(fh.euc_dist() / fw.euc_dist())
                lf_fh_index.append(lf.euc_dist() / fh.euc_dist())
                mw_fw_index.append(mw.euc_dist() / fw.euc_dist())
                mw_fh_index.append(mw.euc_dist() / fh.euc_dist())

        metrics = np.array([facial_index,
                            lf_fh_index,
                            mw_fw_index,
                            mw_fh_index])

        return metrics
