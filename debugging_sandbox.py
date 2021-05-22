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

# %% ###### IMPORT pkl file containing data in form of expobj
trial = 't-012'
date = '2021-01-19'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date,
                                           pkl_path='/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl' % (date, date, trial, date, trial))


# %% classifying stims as in or out of seizures
from utils.paq_utils import paq_read, frames_discard

def collect_seizures_info(expobj, seizures_lfp_timing_matarray=None, discard_all=True):
    print('\ncollecting information about seizures...')
    expobj.seizures_lfp_timing_matarray = seizures_lfp_timing_matarray  # path to the matlab array containing paired measurements of seizures onset and offsets

    # retrieve seizure onset and offset times from the seizures info array input
    paq = paq_read(file_path=expobj.paq_path, plot=False)

    # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
    # NOTE: the output of all of the following function is in dimensions of the FRAME CLOCK (not official paq clock time)
    if seizures_lfp_timing_matarray is not None:
        print('-- using matlab array to collect seizures %s: ' % seizures_lfp_timing_matarray)
        bad_frames, expobj.seizure_frames, expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets = frames_discard(
            paq=paq[0], input_array=seizures_lfp_timing_matarray, total_frames=expobj.n_frames,
            discard_all=discard_all)
    else:
        print('-- no matlab array given to use for finding seizures.')
        bad_frames = frames_discard(paq=paq[0], input_array=seizures_lfp_timing_matarray,
                                    total_frames=expobj.n_frames,
                                    discard_all=discard_all)

    print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
    print('|- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])

    if seizures_lfp_timing_matarray is not None:
        # print('|-now creating raw movies for each sz as well (saved to the /Analysis folder) ... ')
        # expobj.subselect_tiffs_sz(onsets=expobj.seizure_lfp_onsets, offsets=expobj.seizure_lfp_offsets,
        #                         on_off_type='lfp_onsets_offsets')

        print('|-now classifying photostims at phases of seizures ... ')
        expobj.stims_in_sz = [stim for stim in expobj.stim_start_frames if stim in expobj.seizure_frames]
        expobj.stims_out_sz = [stim for stim in expobj.stim_start_frames if stim not in expobj.seizure_frames]
        expobj.stims_bf_sz = [stim for stim in expobj.stim_start_frames
                            for sz_start in expobj.seizure_lfp_onsets
                            if 0 < (
                                    sz_start - stim) < 2 * expobj.fps]  # select stims that occur within 2 seconds before of the sz onset
        expobj.stims_af_sz = [stim for stim in expobj.stim_start_frames
                            for sz_start in expobj.seizure_lfp_offsets
                            if 0 < -1 * (
                                    sz_start - stim) < 2 * expobj.fps]  # select stims that occur within 2 seconds afterof the sz offset
        print(' \n|- stims_in_sz:', expobj.stims_in_sz, ' \n|- stims_out_sz:', expobj.stims_out_sz,
              ' \n|- stims_bf_sz:', expobj.stims_bf_sz, ' \n|- stims_af_sz:', expobj.stims_af_sz)
        aoplot.plot_lfp_stims(expobj)
    expobj.save_pkl()

collect_seizures_info(expobj, seizures_lfp_timing_matarray='/home/pshah/mnt/qnap/Analysis/2021-01-19/paired_measurements/2021-01-19_PS07_012.mat',
                      discard_all=False)
#%%

## save downsampled TIFF

stack = aoutils.plot_single_tiff(tiff_path='/home/pshah/mnt/qnap/Data/2021-01-08/2021-01-08_t-007/2021-01-08_t-007_Cycle00001_Ch3_downsampled.tif')


# Downscale images by half​
stack = tf.imread('/home/pshah/mnt/qnap/Data/2021-01-08/2021-01-08_t-007/2021-01-08_t-007_Cycle00001_Ch3_downsampled.tif')

shape = np.shape(stack)

input_size = stack.shape[1]
output_size = 512
bin_size = input_size // output_size
small_image = stack.reshape((shape[0], output_size, bin_size,
                                      output_size, bin_size)).mean(4).mean(2)

plt.imshow(stack[0], cmap='gray'); plt.show()
plt.imshow(small_image[0], cmap='gray'); plt.show()

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