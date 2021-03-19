# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')



# %run ./setup_notebook.ipynb
# print(sys.path)
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
from utils import funcs_pj as pj

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import utils.funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw

###### IMPORT pkl file containing data in form of expobj
trial = 't-011'
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

# %%
sz_csv = '/home/pshah/mnt/qnap/Analysis/2020-12-18/2020-12-18_t-013/2020-12-18_t-013_stim-9222.tif_border.csv'
expobj.classify_cells_sz(sz_border_path=sz_csv, to_plot=True, title='9222', flip=True)
