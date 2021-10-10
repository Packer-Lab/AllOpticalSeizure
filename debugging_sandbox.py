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
to_suite2p = ['t-005', 't-006', 't-007', 't-008', 't-011', 't-012', 't-013', 't-014', 't-016',
              't-017', 't-018', 't-019', 't-020', 't-021']
baseline_trials = ['t-005', 't-006'] # specify which trials to use as spont baseline
# note ^^^ this only works currently when the spont baseline trials all come first, and also back to back


trials = ['t-020']
trial = 't-020'

for trial in trials:
    ###### IMPORT pkl file containing expobj
    date = '2020-12-19'
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, do_processing=False)
    expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-19/suite2p/alloptical-2p-1x-alltrials/plane0'
    expobj.seizures_lfp_timing_matarray = expobj.seizures_info_array
    expobj.collect_seizures_info()

    expobj.pre_stim = int(0.5 * expobj.fps)  # length of pre stim trace collected
    expobj.post_stim = int(3 * expobj.fps)  # length of post stim trace collected
    expobj.post_stim_response_window_msec = 500  # msec
    expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)

    aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=to_suite2p, baseline_trials=baseline_trials,
                                                force_redo=True)