# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#  This class is responsible for calling the statistical analysis script. It
#  will output the results and graphs to the data/ldmks directory
#
# -----------------------------------------------------------------------------
import os
import sys
import rpy2.robjects as robjects


def main_method(mod_path):
    """
    Calls the R script performing statistical analysis.

    Creates a python function from the R system command using rpy2 then
    sources the R script stats.R and passes the path to the data/ldmks/
    directory containing the csv files with the landamark coordinates.

    Parameters
    ----------
    mod_path : string
        Path to the pfla module.

    Returns
    -------
    None
    """

    source_path = os.path.join(mod_path, "fcn", "stats.R")
    data_path = os.path.join(mod_path, "data", "ldmks")
    source_call = "Rscript {} {}".format(source_path, data_path)

    rsource = robjects.r["system"]
    rpaste = robjects.r["paste"]
    rsource(rpaste(source_call))
