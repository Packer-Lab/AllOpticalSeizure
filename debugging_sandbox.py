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
###### IMPORT pkl file containing data in form of expobj
trial = 't-011'
date = '2020-12-18'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date)

# matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
# expobj.paqProcessing()
# expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
# expobj.save()

# aoplot.plotSLMtargetsLocs(expobj, background=None)

# %% CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV


# FRIST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv
if expobj.sz_boundary_csv_done:
    pass
else:
    sys.exit()

# specify stims for classifying cells
on_ = []
# on_ = [expobj.stim_start_frames[0]]  # uncomment if imaging is starting mid seizure
on_.extend(expobj.stims_bf_sz)


# import the CSV file in and classify cells by their location in or out of seizure

expobj.not_flip_stims = []

print('working on classifying cells for stims start frames:')
expobj.slmtargets_sz_stim = {}
for on, off in zip(on_, expobj.stims_af_sz):
    stims_of_interest = [stim for stim in expobj.stim_start_frames if on <= stim <= off]
    print('|-', stims_of_interest)

    for stim in stims_of_interest:
        sz_border_path = "%s/boundary_csv/%s_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path, expobj.metainfo['date'], trial, stim)
        assert os.path.exists(sz_border_path)
        if stim in expobj.not_flip_stims:
            flip = False
        else:
            flip = True

        in_sz, out_sz = expobj.classify_slmtargets_sz_bound(sz_border_path, to_plot=True, title='%s' % stim, flip=flip)
        expobj.slmtargets_sz_stim[stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure
