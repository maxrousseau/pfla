# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#
#  This class is responsible for calling the statistical analysis script. It 
#  will output the results and graphs to the data/ldmks directory
#
#------------------------------------------------------------------------------

import rpy2.robjects as robjects
import sys

for p in sys.path:
    if 'packages' in p:
        mod_path = p
source_path = mod_path + "/pfla/fcn/stats.R"
data_path = mod_path + "/pfla/data/ldmks/"
source_call = "Rscript " + source_path + " " + data_path

def main():
    """
    Calls the R script performing statistical analysis.

    Creates a python function from the R system command using rpy2 then
    sources the R script stats.R and passes the path to the data/ldmks/
    directory containing the csv files with the landamark coordinates.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    rsource = robjects.r["system"]
    rpaste = robjects.r["paste"]
    rsource(rpaste(source_call))

