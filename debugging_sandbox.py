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

x_size=100
y_size=100
t_size=100

# dummy = np.asarray([np.array([[i]*x_size]*y_size) for i in range(t_size)])
time = np.asarray([np.array([[i]*x_size]*y_size) for i in range(t_size)])

x = np.asarray(list(range(x_size))*y_size*t_size)
y = np.asarray([i_y for i_y in range(y_size) for i_x in range(x_size)] * t_size)
z = time.flatten()

im_array = np.array([x, y, z], dtype=np.float)

assert len(x) == len(y) == len(z), print('length mismatch between x{len(x)}, y{len(y)}, and z{len(z)}')

# plot 3D projection scatter plot
fig = plt.figure(figsize=[10,4])
ax = plt.axes(projection='3d')
ax.scatter(im_array[2], im_array[1], im_array[0], c=im_array[2], cmap='Oranges', linewidth=0.5)

# ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.grid(False)

ax.set_xlabel('time (frames)')
ax.set_ylabel('y axis')
ax.set_zlabel('x axis')
# fig.patch.set_facecolor('black')

fig.show()
