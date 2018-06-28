# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#
#  This class is responsible for calling the statistical analysis script. It 
#  will output the results and graphs to the data/ldmks directory
#
#------------------------------------------------------------------------------

import rpy2.robjects as robjects

def main():
    """Calls the R script performing statistical analysis

    Description:
        creates a python function [rsource] from the R source command using rpy2 then
        sources [rsource](/path/to/script)

    """
    rsource = robjects.r["source"]
    rsource("fcn/stats.R")

