# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#  This class is responsible for calling the statistical analysis script. It
#  will output the results and graphs to the data/ldmks directory
#
# -----------------------------------------------------------------------------
import os
import rpy2.robjects as robjects

mod_path = os.path.dirname(os.path.abspath(__file__))


def main_method(landmark_path):
    """Calls the R script performing statistical analysis.

    Creates a python function from the R system command using rpy2 then
    sources the R script stats.R and passes the path to the data/ldmks/
    directory containing the csv files with the landamark coordinates.

    Parameters
    ----------
    landmark_path : string
        Path to the landmark.
    """

    source_path = os.path.join(mod_path, "stats.R")
    source_call = "Rscript {} {}".format(source_path, landmark_path)

    rsource = robjects.r["system"]
    rpaste = robjects.r["paste"]
    rsource(rpaste(source_call))
