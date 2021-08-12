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
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

# %% TODO plot trial-averaged photostimulation response dFF curves for all experiments - broken down by pre-4ap, outsz and insz (excl. sz bound)

def plot_periphotostim_avg(arr, stim_duration, fps, exp_prestim, pre_stim=1.0, post_stim=3.0, title='', y_lims=None, expobj=None,
                           x_label=None, y_label=None, **kwargs):
    """
    plot trace across all stims
    :param arr:
    :param stim_duration: seconds of stimulation duration
    :param exp_prestim: frames of pre-stim data collected for each trace for this expobj (should be under expobj.pre_stim
    :param pre_stim: seconds of array to plot for pre-stim period
    :param post_stim: seconds of array to plot for post-stim period
    :param title:
    :param y_lims:
    :param x_label:
    :param y_label:
    :param kwargs:
        options include:
            'edgecolor': str, edgecolor of the individual traces behind the mean trace
            'savepath': str, path to save plot to
            'show': bool = to show the plot or not
    :return:
    """
    x = list(range(arr.shape[1]))

    len_ = len(arr)
    flu_avg = np.mean(arr, axis=0)

    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=[8, 6])


    ax.margins(0)
    # ax.axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, alpha=0.2, color='tomato')
    ax.axvspan(exp_prestim, exp_prestim + int(stim_duration*fps), alpha=0.2, color='tomato')
    for cell_trace in arr:
        if 'edgecolor' in kwargs.keys():
            ax.plot(x, cell_trace, linewidth=1, alpha=0.6, c=kwargs['edgecolor'], zorder=1)
        else:
            ax.plot(x, cell_trace, linewidth=1, alpha=0.5, zorder=1)
    ax.plot(x, flu_avg, color='black', linewidth=2.3, zorder=2)  # plot average trace
    ax.set_ylim(y_lims)
    if pre_stim and post_stim:
        ax.set_xlim(exp_prestim - int(pre_stim * fps), exp_prestim + int(stim_duration*fps) + int(post_stim * fps) + 1)

    # # change x axis ticks to seconds
    # if 'time' in x_label or 'Time' in x_label:
    #     label_format = '{:,.2f}'
    #     labels = [item for item in ax.get_xticks()]
    #     for item in labels:
    #         labels[labels.index(item)] = round(item / expobj.fps, 2)
    #     ax.set_xticklabels([label_format.format(x) for x in labels])

    # change x axis ticks to seconds
    if 'Time' in x_label or 'time' in x_label:
        labels = list(np.linspace(0, int(arr.shape[1] / fps), 7))  # x axis tick label every 500 msec
        labels = [round(label,1) for label in labels]
        ax.set_xticks(ticks=[(label * fps) for label in labels])
        ax.set_xticklabels(labels)
        ax.set_xlabel('Time (secs)')

        if 'post' in x_label and 'stimulation' in x_label:
            labels = list(np.linspace(-pre_stim, stim_duration + post_stim, 7))  # x axis tick label every 500 msec
            labels = [round(label, 1) for label in labels]
            labels_locs = np.linspace(exp_prestim - int(pre_stim * fps), exp_prestim + int(stim_duration*fps) + int(post_stim * fps), 7)
            ax.set_xticks(ticks=[int(label) for label in labels_locs])
            ax.set_xticklabels(labels)
            ax.set_xlabel('Time post stimulation (secs)')
        else:
            ax.set_xlabel('Frames')
    else:
        ax.set_xlabel('Frames')

        # labels = [item for item in ax.get_xticks()]
        # for item in labels:
        #     labels[labels.index(item)] = int(round(item / expobj.fps))
        # ax.set_xticklabels(labels)
        # ax.set_xlabel('Time (secs.)')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if 'savepath' in kwargs.keys():
        plt.savefig(kwargs['savepath'])

    # if 'show' in kwargs.keys():
    #     if kwargs['show'] is True:
    #         plt.show()
    # else:
    #     plt.show()

    # finalize plot, set title, and show or return axes
    if 'fig' in kwargs.keys():
        ax.title.set_text((title + ' - %s' % len_ + ' traces'))
        return fig, ax
    else:
        ax.set_title((title + ' - %s' % len_ + ' traces'), horizontalalignment='center', verticalalignment='top',
                     pad=60,
                     fontsize=10, wrap=True)
    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            plt.show()
        else:
            pass
    else:
        plt.show()



dffTraces = []
for i in allopticalResults.pre_4ap_trials:
    pass
i = allopticalResults.pre_4ap_trials[0]
prep = i[0][:-6]
trial = i[0][6:]
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep)

# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
x = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg])
y_label = 'pct. dFF (normalized to prestim period)'
# y_label = 'dFstdF (normalized to prestim period)'

plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dffAvg, fps=expobj.fps, stim_duration=expobj.stim_dur/1000, pre_stim=0.25, exp_prestim=expobj.pre_stim,
                              post_stim=2.75, title=(experiment + '- responses of all photostim targets'),
                              figsize=[5, 4], y_label=y_label, x_label='Time post stimulation (secs)')
