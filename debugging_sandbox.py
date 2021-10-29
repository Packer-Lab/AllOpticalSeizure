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
def plot_bar_with_points(data, title='', x_tick_labels=[], legend_labels: list = [], points: bool = True, bar: bool = True, colors: list = ['black'], ylims=None, xlims=True, text_list=None,
                         x_label=None, y_label=None, alpha=0.2, savepath=None, expand_size_x=1, expand_size_y=1, shrink_text: float = 1, show_legend=False,
                         paired=False, **kwargs):
    """
    all purpose function for plotting a bar graph of multiple categories with the option of individual datapoints shown
    as well. The individual datapoints are drawn by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph. The individual points can also be paired in which case they will be centered. The bar can also be turned off.

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
    :param text_list: list of text to add to each category of data on the plot
    :param text_shift: float; number between 0.5 to 1 used to adjust precise positioning of the text in text_list
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :param expand_size_x: factor to use for expanding figure size
    :param expand_size_y: factor to use for expanding figure size
    :param paired: bool, if True then draw lines between data points of the same index location in each respective list in the data
    :return: matplotlib plot
    """

    # collect some info about data to plot
    w = 0.3  # mean bar width
    x = list(range(len(data)))
    y = data
    if len(colors) != len(x):
        colors = colors * len(x)


    # initialize plot
    if 'fig' in kwargs.keys():
        f = kwargs['fig']
        ax = kwargs['ax']
    else:
        f, ax = plt.subplots(figsize=((5 * len(x) / 2) * expand_size_x, 3 * expand_size_y))

    if paired:
        assert len(x) > 1

    # start making plot
    if not bar:
        for i in x:
            ## plot the mean line
            ax.plot(np.linspace(x[i] * w * 2.5 - w / 2, x[i] * w * 2.5 + w / 2, 3), [np.mean(y[i])] * 3, color='black')
        lw = 0,
        edgecolor = None
        # since no bar being shown on plot (lw = 0 from above) then use it to plot the error bars
        ax.bar([x * w * 2.5 for x in x],
               height=[np.mean(yi) for yi in y],
               yerr=[np.std(yi, ddof=1) for yi in y],  # error bars
               capsize=4.5,  # error bar cap width in points
               width=w,  # bar width
               linewidth=lw,  # width of the bar edges
               edgecolor=edgecolor,
               color=(0, 0, 0, 0),  # face edgecolor transparent
               zorder=2
               )
    elif bar:
        if 'edgecolor' not in kwargs.keys():
            edgecolor = 'black',
            lw = 1
        else:
            edgecolor = kwargs['edgecolor'],
            lw = 1
        # plot bar graph
        ax.bar([x * w * 2.5 for x in x],
               height=[np.mean(yi) for yi in y],
               # yerr=np.asarray([np.asarray([0, np.std(yi, ddof=1)]) for yi in y]).T,  # error bars
               capsize=4.5,  # error bar cap width in points
               width=w,  # bar width
               linewidth=lw,  # width of the bar edges
               edgecolor=edgecolor,
               color=(0, 0, 0, 0),  # face edgecolor transparent
               zorder=2
               )
        ax.errorbar([x * w * 2.5 for x in x], [np.mean(yi) for yi in y], fmt='none', yerr = np.asarray([np.asarray([0, np.std(yi, ddof=1)]) for yi in y]).T, ecolor='gray',
                    capsize=5, zorder=0)
    else:
        ReferenceError('something wrong happened with the bar bool parameter...')

    ax.set_xticks([x * w * 2.5 for x in x])
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([(x[0] * w * 2) - w * 1.20, (x[-1] * w * 2.5) + w * 1.20])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims_ = [-1, 1]
        ax.set_xlim(xlims_)

    if len(legend_labels) == 0:
        if len(x_tick_labels) == 0:
            x_tick_labels = [None] * len(x)
        legend_labels = x_tick_labels

    if points:
        if not paired:
            for i in x:
                # distribute scatter randomly across whole width of bar
                ax.scatter(x[i] * w * 2.5 + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha, label=legend_labels[i])

        else:  # connect lines to the paired scatter points in the list
            if len(x) > 0:
                for i in x:
                    # plot points  # dont scatter location of points if plotting paired lines
                    ax.scatter([x[i] * w * 2.5] * len(y[i]), y[i], color=colors[i], alpha=0.5,
                               label=legend_labels[i], zorder=3)
                for i in x[:-1]:
                    for point_idx in range(len(y[i])):  # draw the lines connecting pairs of data
                        ax.plot([x[i] * w * 2.5 + 0.058, x[i+1] * w * 2.5 - 0.048], [y[i][point_idx], y[i+1][point_idx]], color='black', zorder=2, alpha=alpha)

                # for point_idx in range(len(y[i])):  # slight design difference, with straight line going straight through the scatter points
                #     ax.plot([x * w * 2.5 for x in x],
                #             [y[i][point_idx] for i in x], color='black', zorder=0, alpha=alpha)

            else:
                ReferenceError('cannot do paired scatter plotting with only one data category')

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=10)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(x_label, fontsize=8*shrink_text)
    ax.set_ylabel(y_label, fontsize=12*shrink_text)
    if savepath:
        plt.savefig(savepath)
    if len(x) > 1:
        plt.xticks(rotation=45)
        # plt.setp(ax.get_xticklabels(), rotation=45)

    # add text to the figure if given:
    if text_list:
        assert len(x) == len(text_list), 'please provide text_list of same len() as data'
        if 'text_shift' in kwargs.keys():
            text_shift = kwargs['text_shift']
        else:
            text_shift = 0.8
        if 'text_y_pos' in kwargs.keys():
            text_y_pos = kwargs['text_y_pos']
        else:
            text_y_pos = max([np.percentile(y[i], 95) for i in x])
        for i in x:
            ax.text(x[i] * w * 2.5 - text_shift*w / 2, text_y_pos, text_list[i]),

    if len(legend_labels) > 1:
        if show_legend:
            ax.legend(bbox_to_anchor=(1.01, 0.90), fontsize=8*shrink_text)

    # add title
    if 'fig' not in kwargs.keys():
        ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=25,
                     fontsize=12*shrink_text, wrap=True)
    else:
        ax.set_title((title),horizontalalignment='center', verticalalignment='top',
                     fontsize=12*shrink_text, wrap=True)

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            # Tweak spacing to prevent clipping of ylabel
            # f.tight_layout()
            f.show()
        else:
            return f, ax
    else:
        # Tweak spacing to prevent clipping of ylabel
        # f.tight_layout()
        f.show()

cols = ['pre4ap_pos', 'post4ap_pos']
data = [np.random.random(10), np.random.random(10)]
plot_bar_with_points(data, x_tick_labels=cols, colors=['skyblue', 'steelblue'],
                        bar=True, paired=True, expand_size_x=0.6, expand_size_y=1.3, title='# of Positive responders',
                        y_label='# of sig. responders')