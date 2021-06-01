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

#%%

trial = 't-016'
date = '2020-12-19'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)

#%%
def plot_bar_with_points(data, title='', x_tick_labels=[], legend_labels: list = [], points=True, bar=True, colors=['black'], ylims=None, xlims=None,
                         x_label=None, y_label=None, alpha=0.2, savepath=None, expand_size_x=1, expand_size_y=1, shrink_text: float = 1, **kwargs):
    """
    general purpose function for plotting a bar graph of multiple categories with the individual datapoints shown
    as well. The latter is achieved by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph.

    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param legend_labels:
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :param expand_size_x: factor to use for expanding figure size
    :param expand_size_y: factor to use for expanding figure size
    :return: matplotlib plot
    """

    # collect some info about data to plot
    w = 0.3  # bar width
    x = list(range(len(data)))
    y = data
    if len(colors) != len(x):
        colors = colors * len(x)


    # initialize plot
    if 'fig' in kwargs.keys():
        f = kwargs['fig']
        ax = kwargs['ax']
        legend_labels = x_tick_labels
        assert len(legend_labels) > 0
    else:
        if len(legend_labels) > 0:
            f, ax = plt.subplots(figsize=((5 * len(x) / 2) * expand_size_x, 3 * expand_size_y))
        else:
            f, ax = plt.subplots(figsize=((2 * len(x) / 2) * expand_size_x, 3 * expand_size_y))
            legend_labels = x_tick_labels
            assert len(legend_labels) > 0


    # start making plot
    if not bar:
        for i in x:
            # ax.plot(np.linspace(x[i] - w / 2, x[i] + w / 2, 3), [np.mean(yi) for yi in y] * 3, edgecolor=colors[i])
            ax.plot(np.linspace(x[i] * w * 2 - w / 2, x[i] * w * 2 + w / 2, 3), [np.mean(y[i])] * 3, color='black')
        lw = 0,
        edgecolor = None
    else:
        edgecolor = 'black',
        lw = 1

    # plot bar graph, or if no bar (when lw = 0 from above) then use it to plot the error bars
    ax.bar([x * w * 2 for x in x],
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y],  # error bars
           capsize=4.5,  # error bar cap width in points
           width=w,  # bar width
           linewidth=lw,  # width of the bar edges
           # tick_label=x_tick_labels,
           edgecolor=edgecolor,
           color=(0, 0, 0, 0),  # face edgecolor transparent
           )
    ax.set_xticks([x * w * 2 for x in x])
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([xlims[0] - 2 * w, xlims[1] + 2 * w])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims = [-1, 1]
        ax.set_xlim(xlims)

    if points:
        for i in x:
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] * w * 2 + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha, label=legend_labels[i])

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=10)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(x_label, fontsize=8*shrink_text)
    ax.set_ylabel(y_label, fontsize=8*shrink_text)
    if savepath:
        plt.savefig(savepath)
    if len(x) > 1:
        plt.xticks(rotation=45)
        # plt.setp(ax.get_xticklabels(), rotation=45)

    if len(legend_labels) > 1:
        ax.legend(bbox_to_anchor=(1.01, 0.90), fontsize=8*shrink_text)

    # add title
    if 'fig' not in kwargs.keys():
        ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=25,
                     fontsize=8*shrink_text, wrap=True)
    else:
        ax.title.set_text((title))

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            # Tweak spacing to prevent clipping of ylabel
            f.tight_layout()
            f.show()
        else:
            return f, ax
    else:
        # Tweak spacing to prevent clipping of ylabel
        f.tight_layout()
        f.show()



# histogram density plot with gaussian best fit line
def plot_hist_density(data, colors: list = None, fill_color: list = None, **kwargs):

    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=[20, 3])

    if len(data) == 1:
        colors = ['black']
        fill_color = ['steelblue']
    else:
        assert len(data) == len(colors)
        assert len(data) == len(fill_color)

    for i in range(len(data)):
        # the histogram of the data
        num_bins = 10
        n, bins, patches = ax.hist(data[i], num_bins, density=1, alpha=0.2)  # histogram hidden currently

        # add a 'best fit' line
        mu = np.mean(data[i])  # mean of distribution
        sigma = np.std(data[i])  # standard deviation of distribution

        x = np.linspace(bins[0], bins[-1], 50)
        y1 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (x - mu))**2))
        ax.plot(x, y1, linewidth=2, c=colors[i], zorder=2)
        ax.fill_between(x, y1, color=fill_color[i], zorder=2)
        if 'x_label' in kwargs:
            ax.set_xlabel(kwargs['x_label'])
        if 'y_label' in kwargs:
            ax.set_ylabel(kwargs['y_label'])
        else:
            ax.set_ylabel('Probability density')


    # add title
    if 'fig' not in kwargs.keys():
        if 'title' in kwargs:
            ax.set_title(kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))
        else:
            ax.set_title(r'Histogram: $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            # Tweak spacing to prevent clipping of ylabel
            fig.tight_layout()
            fig.show()
        else:
            pass
    else:
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        fig.show()

    if 'fig' in kwargs.keys():
        # adding text because adding title doesn't seem to want to work when piping subplots
        if 'shrink_text' in kwargs.keys():
            shrink_text = kwargs['shrink_text']
        else:
            shrink_text = 1

        ax.title.set_text(kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))
        # ax.text(0.98, 0.97, kwargs['title'] + r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)),
        #         verticalalignment='top', horizontalalignment='right',
        #         transform=ax.transAxes, fontweight='bold',
        #         color='black', fontsize=10 * shrink_text)
        return fig, ax


# make response magnitude and response success rate figure
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=((5 * 4), 5), nrows=1, ncols=4)
# stims out sz
data = [[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]]
fig, ax1 = plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='%s - stims_out_sz - ' % trial, fig=fig, ax=ax1, show=False)
fig, ax2 = plot_bar_with_points(data=[list(expobj.outsz_StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                              ylims=[0, 100], bar=False, y_label='% success stims.', title='target success rate (stims out sz)', expand_size_x=2,
                              show=False, fig=fig, ax=ax3)
# stims in sz
data = [[np.mean(expobj.insz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]]
fig, ax3 = plot_hist_density(data, x_label='response magnitude (dF/stdF)', title='%s stims_in_sz - ' % trial, fig=fig, ax=ax2, show=False)
fig, ax4 = plot_bar_with_points(data=[list(expobj.insz_StimSuccessRate_SLMtargets.values())], x_tick_labels=[trial],
                              ylims=[0, 100], bar=False, y_label='% success stims.', title='target success rate (stims in sz)', expand_size_x=2,
                              show=False, fig=fig, ax=ax4)
fig.show()


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