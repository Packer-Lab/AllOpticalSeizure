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

print('s2p neu. corrected cell traces statistics: ')
for cell in expobj.s2p_cell_nontargets:
    cell_idx = expobj.cell_id.index(cell)
    print('mean: %s   \t min: %s  \t max: %s  \t std: %s' %
          (np.round(np.mean(expobj.raw[cell_idx]), 2), np.round(np.min(expobj.raw[cell_idx]), 2), np.round(np.max(expobj.raw[cell_idx]), 2),
           np.round(np.std(expobj.raw[cell_idx], ddof=1), 2)))