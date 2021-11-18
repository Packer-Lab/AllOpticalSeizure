# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')

import numpy as np
import pandas as pd
from scipy import stats, signal
import statsmodels.api
import statsmodels as sm
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'

allopticalResults = aoutils.AllOpticalResults(save_path=results_object_path)
