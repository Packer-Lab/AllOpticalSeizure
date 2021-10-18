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
def xyloc_responses(expobj, df, label='response magnitude', clim=[-10, +10], plot_target_coords=True, title='', save_fig: str = None):
    """
    plot to show the response magnitude of each cell as the actual's filling in the cell's ROI pixels.

    :param expobj: expobj associated with data
    :param df: pandas dataframe (cell_id x stim frames)
    :param clim: color limits
    :param plot_target_coords: bool, if True plot the actual X and Y coords of all photostim cell targets
    :param save_fig: where to save the save figure (optional)
    :return:
    """
    # stim_timings = [str(i) for i in
    #                 expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

    # if to_plot == 'dfstdf':
    #     average_responses = expobj.dfstdf_all_cells[stim_timings].mean(axis=1).tolist()
    # elif to_plot == 'dff':
    #     average_responses = expobj.dff_responses_all_cells[stim_timings].mean(axis=1).tolist()
    # else:
    #     raise Exception('need to specify to_plot arg as either dfstdf or dff in string form!')

    cells_ = list(df.index)
    average_responses = df.mean(axis=1).tolist()
    # make a matrix containing pixel locations and responses at each of those pixels
    responses = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')

    for n in cells_:
        idx = expobj.cell_id.index(n)
        ypix = expobj.stat[idx]['ypix']
        xpix = expobj.stat[idx]['xpix']
        responses[ypix, xpix] = 100. + 1 * round(average_responses[cells_.index(n)], 2)

    # mask some 'bad' data, in your case you would have: data < 0.005
    responses = np.ma.masked_where(responses < 0.005, responses)
    cmap = plt.cm.bwr
    cmap.set_bad(color='black')

    plt.figure(figsize=(7, 7))
    im = plt.imshow(responses, cmap=cmap)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(label)

    plt.clim(100+clim[0], 100+clim[1])
    if plot_target_coords:
        for (x, y) in expobj.target_coords_all:
            plt.scatter(x=x, y=y, edgecolors='green', facecolors='none', linewidths=1.0)
    plt.suptitle(title + ' - SLM targets in green', y=0.90, fontsize=10)
    # pj.plot_cell_loc(expobj, cells=expobj.s2p_cell_targets, background_transparent=True)
    plt.show()
    if save_fig is not None:
        plt.savefig(save_fig)

expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')

s_ = np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)
arr = expobj.post_array_responses[expobj.sig_units, :][s_]
df = pd.DataFrame(arr, index=[expobj.s2p_cell_nontargets[i] for i in s_[0]], columns=expobj.stim_start_frames)
xyloc_responses(expobj, df=df, clim=[-0.3, +0.3], plot_target_coords=True)
