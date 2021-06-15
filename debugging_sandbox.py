# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')


# %run ./setup_notebook.ipynb
# print(sys.path)

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys

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
from matplotlib.colors import LinearSegmentedColormap, ColorConverter

trials = ['t-009']

v_lims_periphotostim_heatmap=[-0.1*100, 0.2*100]
zero_point = abs(v_lims_periphotostim_heatmap[0]/v_lims_periphotostim_heatmap[1])
c = ColorConverter().to_rgb
bwr_custom = pj.make_colormap([c('blue'), c('white'), zero_point - 0.20, c('white'), c('red')])

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    date = '2020-12-18'
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL108/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)

    aoutils.slm_targets_responses(expobj, experiment, trial, y_spacing_factor=4, smooth_overlap_traces=5,
                                  figsize=[30, 20], force_redo=True,
                                  linewidth_overlap_traces=0.2, y_lims_periphotostim_trace=[-0.1*100, 0.6*100],
                                  v_lims_periphotostim_heatmap=v_lims_periphotostim_heatmap, cmap=bwr_custom,
                                  save_results=False)



# %%
from matplotlib.colors import LinearSegmentedColormap, ColorConverter

v_lims_periphotostim_heatmap=[-0.2*100, 0.6*100]
zero_point = abs(v_lims_periphotostim_heatmap[0]/v_lims_periphotostim_heatmap[1])
c = ColorConverter().to_rgb
bwr_custom = pj.make_colormap([c('blue'), c('white'), zero_point - 0.08, c('white'), c('red')])

gradient = np.linspace(v_lims_periphotostim_heatmap[0], v_lims_periphotostim_heatmap[1], 256)
gradient = np.vstack((gradient, gradient))
plt.figure(figsize=(10, 3))
plt.imshow(gradient, aspect='auto', cmap=bwr_custom)
cbar = plt.colorbar(boundaries=np.linspace(v_lims_periphotostim_heatmap[0], v_lims_periphotostim_heatmap[1], 1000), ticks=[v_lims_periphotostim_heatmap[0], 0, v_lims_periphotostim_heatmap[1]])
plt.show()


#%%

# ## save downsampled TIFF
#
# stack = aoutils.plot_single_tiff(tiff_path='/home/pshah/mnt/qnap/Data/2021-01-08/2021-01-08_t-007/2021-01-08_t-007_Cycle00001_Ch3_downsampled.tif')
#
#
# # Downscale images by half​
# stack = tf.imread('/home/pshah/mnt/qnap/Data/2021-01-08/2021-01-08_t-007/2021-01-08_t-007_Cycle00001_Ch3_downsampled.tif')
#
# shape = np.shape(stack)
#
# input_size = stack.shape[1]
# output_size = 512
# bin_size = input_size // output_size
# small_image = stack.reshape((shape[0], output_size, bin_size,
#                                       output_size, bin_size)).mean(4).mean(2)
#
# plt.imshow(stack[0], cmap='gray'); plt.show()
# plt.imshow(small_image[0], cmap='gray'); plt.show()

# %%
# original = '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file021_chan0.tif'
# recreated = '/home/pshah/mnt/qnap/Analysis/2021-01-10/2021-01-10_t-008/reg_tiff_t-008.tif'
#
# with tf.TiffFile(original, multifile=False) as input_tif:
#     data_original = input_tif.asarray()
#     print('shape of tiff: ', data_original.shape)
#
# with tf.TiffFile(recreated, multifile=False) as input_tif:
#     data_recreated = input_tif.asarray()
#     print('shape of tiff: ', data_recreated.shape)
#     data_recreated1 = data_recreated[0]
#

# sorted_paths = ['/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file021_chan0.tif',
#                 '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file022_chan0.tif',
#                 '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file023_chan0.tif',
#                 '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file024_chan0.tif',
#                 '/home/pshah/mnt/qnap/Analysis/2021-01-10/suite2p/alloptical-2p-08x-alltrials-reg_tiff/plane0/reg_tif/file025_chan0.tif']
#
# def make_tiff_stack(sorted_paths: list, save_as: str):
#     """
#     read in a bunch of tiffs and stack them together, and save the output as the save_as
#
#     :param sorted_paths: list of string paths for tiffs to stack
#     :param save_as: .tif file path to where the tif should be saved
#     """
#
#     num_tiffs = len(sorted_paths)
#     print('working on tifs to stack: ', num_tiffs)
#
#     with tf.TiffWriter(save_as, bigtiff=True) as tif:
#         for i, tif_ in enumerate(sorted_paths):
#             with tf.TiffFile(tif_, multifile=True) as input_tif:
#                 data = input_tif.asarray()
#                 for frame in data:
#                     tif.write(frame, contiguous=True)
#
#                 # tif.save(data[0])
#             msg = ' -- Writing tiff: ' + str(i + 1) + ' out of ' + str(num_tiffs)
#             print(msg, end='\r')
#             # tif.save(data)
#
# make_tiff_stack(sorted_paths=sorted_paths, save_as='/home/pshah/mnt/qnap/Analysis/2021-01-10/2021-01-10_t-008/reg_tiff_t-008.tif')
#
# # series0 = np.random.randint(0, 255, (32, 32, 3), 'uint8')
# # series1 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
# series0 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
# series1 = np.random.randint(0, 1023, (4, 256, 256), 'uint16')
# tf.imwrite('temp.tif', series0, photometric='minisblack')
# tf.imwrite('temp.tif', series1, append=True)
#
# img = tf.imread('temp.tif')