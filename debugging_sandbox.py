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

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/'


########

prep='RL108'
trial='t-013'
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

# aoplot.plot_lfp_stims(expobj, xlims=[0.2e7, 1.0e7], linewidth=1.0)


# %%
# plots the raw trace for the Flu mean of the FOV
def plotLfpSignal(expobj, stim_span_color='powderblue', downsample: bool = True, stim_lines: bool = True, sz_markings: bool = False,
                  title='LFP trace', x_axis='time', hide_xlabel=False, **kwargs):
    """make plot of LFP with also showing stim locations
    NOTE: ONLY PLOTTING LFP SIGNAL CROPPED TO 2P IMAGING FRAME START AND END TIMES - SO CUTTING OUT THE LFP SIGNAL BEFORE AND AFTER"""

    # if there is a fig and ax provided in the function call then use those, otherwise start anew
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=[60 * (expobj.stim_start_times[-1] + 1e5 - (expobj.stim_start_times[0] - 1e5)) / 1e7, 3])

    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    else:
        alpha = 1

    # plot LFP signal
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'steelblue'

    # option for downsampling of data plot trace
    x = range(len(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual]))
    signal = expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual]
    if downsample:
        labels = list(range(0, int(len(signal) / expobj.paq_rate * 1), 30))[::2]
        down = 1000
        signal = signal[::down]
        x = x[::down]
        assert len(x) == len(signal), print('something went wrong with the downsampling')

    # change linewidth
    if 'linewidth' in kwargs:
        lw = kwargs['linewidth']
    else:
        lw = 0.4

    ax.plot(x, signal, c=color, zorder=1, linewidth=lw, alpha=alpha)  ## NOTE: ONLY PLOTTING LFP SIGNAL RELATED TO
    ax.margins(0)

    # plot stims
    if stim_span_color != '':
        for stim in expobj.stim_start_times:
            stim = (stim - expobj.frame_start_time_actual)
            ax.axvspan(stim - 8, 1 + stim + expobj.stim_duration_frames / expobj.fps * expobj.paq_rate, color=stim_span_color, zorder=1, alpha=0.5)
    else:
        if stim_lines:
            for line in expobj.stim_start_times:
                line = (line - expobj.frame_start_time_actual)
                ax.axvline(x=line+2, color='black', linestyle='--', linewidth=0.6, zorder=0)

    # plot seizure onset and offset markings
    if sz_markings:
        if hasattr(expobj, 'seizure_lfp_onsets'):
            for sz_onset in expobj.seizure_lfp_onsets:
                ax.axvline(x=expobj.frame_clock_actual[sz_onset] - expobj.frame_start_time_actual, color='black', linestyle='--', linewidth=1.0, zorder=0)
            for sz_offset in expobj.seizure_lfp_offsets:
                ax.axvline(x=expobj.frame_clock_actual[sz_offset] - expobj.frame_start_time_actual, color='black', linestyle='--', linewidth=1.0, zorder=0)

    # change x axis ticks to seconds
    if 'time' in x_axis or 'Time' in x_axis:
        # set x ticks at every 30 seconds
        # labels = ls(range(0, int(len(signal) / expobj.paq_rate * down), 30))[::2]
        # print('x_axis labels: ', labels)
        ax.set_xticks(ticks=[(label * expobj.paq_rate) for label in labels])
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='both', length=3)
        if not hide_xlabel:
            ax.set_xlabel('Time (secs)')

        # label_format = '{:,.2f}'
        # labels = [item for item in ax.get_xticks()]
        # for item in labels:
        #     labels[labels.index(item)] = int(round(item / expobj.paq_rate, 2))
        # ticks_loc = ax.get_xticks().tolist()
        # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # ax.set_xticklabels([label_format.format(x) for x in labels])
        # ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('paq clock')
    ax.set_ylabel('Voltage')
    # ax.set_xlim([expobj.frame_start_time_actual, expobj.frame_end_time_actual])  ## this should be limited to the 2p acquisition duration only

    # set ylimits:
    if 'ylims' in kwargs:
        ax.set_ylim(kwargs['ylims'])
    else:
        ax.set_ylim([np.mean(expobj.lfp_signal) - 10, np.mean(expobj.lfp_signal) + 10])

    # add title
    if not 'fig' in kwargs.keys():
        ax.set_title(
            '%s - %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))

    # options for showing plot or returning plot
    if 'show' in kwargs.keys():
        plt.show() if kwargs['show'] else None
    else:
        plt.show()


    return fig, ax if 'fig' in kwargs.keys() else None

plotLfpSignal(expobj, downsample=True, figsize=(10,3), x_axis='paq clock')