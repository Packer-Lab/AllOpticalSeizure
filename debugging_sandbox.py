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

pre_frames = 15*5
post_frames = 15*10
test_frames = int(15 * 0.5)

s = np.s_[pre_frames - test_frames: pre_frames]

a = np.random.random([100,100,100])
a_ = a[:, s, :]

a = np.arange(10).reshape(2, 5)
m = np.s_[3:4]
a[:, m]