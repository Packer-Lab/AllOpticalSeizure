# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')


# %run ./setup_notebook.ipynb
# print(sys.path)

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
# sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf


########
# %%
# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

i = allopticalResults.post_4ap_trials[0]
j = 0
prep = 'RL109'
trial = 't-016'
print('\nprogress @ ', prep, trial)
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

redo_processing = True  # flag to use when rerunning this whole for loop multiple times
if 'post' in expobj.metainfo['exptype']:
    if redo_processing:
        aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials,
                                                    baseline_trials=expobj.baseline_trials,
                                                    plots=False, force_redo=False)

