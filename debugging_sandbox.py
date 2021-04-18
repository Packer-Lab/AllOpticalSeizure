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
import alloptical_plotting as aoplot
import utils.funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf

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


########

###### IMPORT pkl file containing data in form of expobj
trial = 't-013'
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

# aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3], show=True)

# aoplot.plotLfpSignal(expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])

aoplot.plotImgSLMtargetsLocs(expobj, background=expobj.meanFluImg_registered)

#%%

# testing ordering suite2p cells in order of recruitment into seizure

sz = 2
sz_onset, sz_offset = expobj.stims_bf_sz[sz], expobj.stims_af_sz[sz+1]
expobj.dff_flu = aoutils.normalize_dff_baseline(expobj.raw, baseline_array=expobj)
dff_mean = [np.percentile(trace, 50) for trace in expobj.dff_flu]

x = expobj.dff_flu[[expobj.cell_id.index(cell) for cell in expobj.good_cells], sz_onset:sz_offset]
x_mean = [np.percentile(trace, 50) for trace in x]


## TODO organize individual cells in array in order of peak firing rate
## - calculate some measure of peak firing for each cell, then determine where the derivative occurs earliest (get that index from the array)
## - assign the index to each trace index, and then organize the trace indexes in that order. then finally re-order the indexes in accordance with the new indexes.
stims = [(stim - sz_onset) for stim in expobj.stim_start_frames if sz_onset <= stim < sz_offset]
stims_off = [(stim + expobj.stim_duration_frames - 1) for stim in stims]

x_bf = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_bf_sz[sz])[0][0]]
x_af = expobj.stim_times[np.where(expobj.stim_start_frames == expobj.stims_af_sz[sz+1])[0][0]]


lfp_signal = expobj.lfp_signal[x_bf:x_af]

aoplot.plot_traces_heatmap(x, stim_on=stims, stim_off=stims_off, cmap='Spectral_r', figsize=(10,6),
                           title=('%s - seizure %s' % (trial, sz)), xlims=None, vmin=100, vmax=500,
                           lfp_signal=lfp_signal)

# -- trying approach to find top 10% signal, and find earliest index above this threshold
x_90 = [np.percentile(trace, 90) for trace in x]


# %%
# expobj.raw_traces_from_targets()
# expobj.save()
#
# def get_alltargets_stim_traces_norm(expobj, targets_idx=None, subselect_cells=None, pre_stim=15, post_stim=200):
#     """
#     primary function to measure the dFF traces for photostimulated targets.
#     :param expobj:
#     :param normalize_to: str; either "baseline" or "pre-stim"
#     :param pre_stim: number of frames to use as pre-stim
#     :param post_stim: number of frames to use as post-stim
#     :return: lists of individual targets dFF traces, and averaged targets dFF over all stims for each target
#     """
#     stim_timings = expobj.stim_start_frames
#
#     if subselect_cells:
#         num_cells = len(expobj.SLMTargets_stims_raw[subselect_cells])
#         targets_trace = expobj.SLMTargets_stims_raw[subselect_cells]
#     else:
#         num_cells = len(expobj.SLMTargets_stims_raw)
#         targets_trace = expobj.SLMTargets_stims_raw
#
#     # collect photostim timed average dff traces of photostim targets
#     targets_dff = np.zeros([num_cells, len(expobj.stim_start_frames), pre_stim + post_stim])
#     targets_dff_avg = np.zeros([num_cells, pre_stim + post_stim])
#
#     targets_dfstdF = np.zeros([num_cells, len(expobj.stim_start_frames), pre_stim + post_stim])
#     targets_dfstdF_avg = np.zeros([num_cells, pre_stim + post_stim])
#
#     SLMTargets_stims_raw = np.zeros([num_cells, len(expobj.stim_start_frames), pre_stim + post_stim])
#     targets_raw_avg = np.zeros([num_cells, pre_stim + post_stim])
#
#     if targets_idx is not None:
#         flu = [targets_trace[targets_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
#                stim not in expobj.seizure_frames]
#
#         # flu_dfstdF = []
#         # flu_dff = []
#         for i in range(len(flu)):
#             trace = flu[i]
#             mean_pre = np.mean(trace[0:pre_stim])
#             trace_dff = ((trace - mean_pre) / mean_pre) * 100
#             std_pre = np.std(trace[0:pre_stim])
#             dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
#
#             SLMTargets_stims_raw[targets_idx, i] = trace
#             targets_dff[targets_idx, i] = trace_dff
#             targets_dfstdF[targets_idx, i] = dFstdF
#         return SLMTargets_stims_raw[targets_idx], targets_dff[targets_idx], targets_dfstdF[targets_idx]
#
#     else:
#         for cell_idx in range(num_cells):
#             # print('considering cell # %s' % cell)
#             flu = [targets_trace[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
#                    stim not in expobj.seizure_frames]
#
#             # flu_dfstdF = []
#             # flu_dff = []
#             for i in range(len(flu)):
#                 trace = flu[i]
#                 mean_pre = np.mean(trace[0:pre_stim])
#                 trace_dff = ((trace - mean_pre) / mean_pre) * 100
#                 std_pre = np.std(trace[0:pre_stim])
#                 dFstdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
#
#                 SLMTargets_stims_raw[cell_idx, i] = trace
#                 targets_dff[cell_idx, i] = trace_dff
#                 targets_dfstdF[cell_idx, i] = dFstdF
#                 # flu_dfstdF.append(dFstdF)
#                 # flu_dff.append(trace_dff)
#
#             # targets_dff.append(flu_dff)  # contains all individual dFF traces for all stim times
#             # targets_dff_avg.append(np.nanmean(flu_dff, axis=0))  # contains the dFF trace averaged across all stim times
#
#             # targets_dfstdF.append(flu_dfstdF)
#             # targets_dfstdF_avg.append(np.nanmean(flu_dfstdF, axis=0))
#
#             # SLMTargets_stims_raw.append(flu)
#             # targets_raw_avg.append(np.nanmean(flu, axis=0))
#
#         targets_dff_avg = np.mean(targets_dff, axis=1)
#         targets_dfstdF_avg = np.mean(targets_dfstdF, axis=1)
#         targets_raw_avg = np.mean(SLMTargets_stims_raw, axis=1)
#
#         return targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, SLMTargets_stims_raw, targets_raw_avg
#
# # targets_dff, targets_dff_avg, targets_dfstdF, targets_dfstdF_avg, SLMTargets_stims_raw, targets_raw_avg = get_alltargets_stim_traces_norm(expobj)
# # array = (np.convolve(SLMTargets_stims_raw[targets_idx], np.ones(w), 'valid') / w)
#
#
# targets_idx = 0
# pre_stim = int(expobj.fps) // 2
# post_stim = 3 * int(expobj.fps)
# SLMTargets_stims_raw, targets_dff, targets_dfstdF = get_alltargets_stim_traces_norm(expobj, targets_idx=targets_idx, pre_stim=pre_stim,
#                                                                            post_stim=post_stim)


# %%
