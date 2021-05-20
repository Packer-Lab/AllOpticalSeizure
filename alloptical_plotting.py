# library of functions that are used for making various plots for all optical photostimulation/imaging experiments
# the data that feeds into these plots are generated by Vape (from alloptical_analysis_photostim.py)

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
import utils.funcs_pj as pj
import alloptical_utils_pj as aoutils
import tifffile as tf
from utils.paq_utils import paq_read, frames_discard


# simple plot of the location of the given cell(s) against a black FOV
def plot_cell_loc(expobj, cells: list, edgecolor: str = '#EDEDED', title=None, background: np.array = None,
                  show_s2p_targets: bool = True, color_float_list: list = None, cmap: str = 'Reds', **kwargs):
    """
    plots an image of the FOV to show the locations of cells given in cells list.
    :param background: either 2dim numpy array to use as the backsplash or None (where black backsplash will be created)
    :param expobj: alloptical or 2p imaging object
    :param edgecolor: str to specify edgecolor of the scatter plot for cells
    :param cells: list of cells to plot
    :param title: str title for plot
    :param color_float_list: if given, it will be used to color the cells according a colormap
    :param cmap: cmap to be used in conjuction with the color_float_array argument
    :param show_s2p_targets: if True, then will prioritize coloring of cell points based on whether they were photostim targets
    :param kwargs: optional arguments
            invert_y: if True, invert the reverse the direction of the y axis
            show: if True, show the plot
            fig: a fig plt.subplots() instance, if provided use this fig for making figure
            ax: a ax plt.subplots() instance, if provided use this ax for plotting
    """

    # if there is a fig and ax provided in the function call then use those, otherwise start anew
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots()

    if background is None:
        black = np.zeros((expobj.frame_x, expobj.frame_y), dtype='uint16')
        ax.imshow(black, cmap='Greys_r', zorder=0)
    else:
        ax.imshow(background)

    x_list = []
    y_list = []
    for cell in cells:
        y, x = expobj.stat[expobj.cell_id.index(cell)]['med']
        x_list.append(x)
        y_list.append(y)

        if show_s2p_targets:
            if hasattr(expobj, 's2p_cell_targets'):
                if cell in expobj.s2p_cell_targets:
                    color_ = '#F02A71'
                else:
                    color_ = 'none'
            else:
                color_ = 'none'
            ax.scatter(x=x, y=y, edgecolors=edgecolor, facecolors=color_, linewidths=0.8)
        elif color_float_list:
            # ax.scatter(x=x, y=y, edgecolors='none', c=color_float_list[cells.index(cell)], linewidths=0.8,
            #            cmap=cmap)
            pass
        else:
            ax.scatter(x=x, y=y, edgecolors=edgecolor, facecolors='none', linewidths=0.8)

    if color_float_list:
        ac = ax.scatter(x=x_list, y=y_list, edgecolors='none', c=color_float_list, linewidths=0.8,
                   cmap=cmap, zorder=1)

        plt.colorbar(ac, ax=ax)

    if background is None:
        ax.set_xlim(0, expobj.frame_x)
        ax.set_ylim(0, expobj.frame_y)

    if title is not None:
        plt.suptitle(title, wrap=True)

    if 'invert_y' in kwargs.keys():
        if kwargs['invert_y']:
            ax.invert_yaxis()

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            fig.show()
        else:
            pass
    else:
        fig.show()

    if 'fig' in kwargs.keys():
        return fig, ax


### plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????
def plot_cell_radius_aspectr(expobj, stat, to_plot, min_vline: int = 4, max_vline: int = 12):
    radius = []
    aspect_ratio = []
    for cell in range(len(stat)):
        # if expobj.cell_id[cell] in expobj.good_cells:
        if expobj.cell_id[cell] in expobj.cell_id:
            radius.append(stat[cell]['radius'])
            aspect_ratio.append(stat[cell]['aspect_ratio'])

    if to_plot == 'radius':
        to_plot_ = radius
        plt.axvline(min_vline / expobj.pix_sz_x, color='grey')
        plt.axvline(max_vline / expobj.pix_sz_x, color='grey')
        n, bins, patches = plt.hist(to_plot_, 100)
        title = 'radius - {%s um to %s um}' % (min_vline, max_vline)
    elif to_plot == 'aspect':
        to_plot_ = aspect_ratio
        n, bins, patches = plt.hist(to_plot_, 100)
        title = 'aspect ratio'

    plt.suptitle('All cells - %s' % title, y=0.95)
    plt.show()
    return to_plot_


### plot the location of all SLM targets, along with option for plotting the mean img of the current trial
def plotSLMtargetsLocs(expobj, background: np.ndarray = None, **kwargs):

    if background is None:
        black = np.zeros((expobj.frame_x, expobj.frame_y), dtype='uint16')
        plt.imshow(black, cmap='gray')
    else:
        plt.imshow(background, cmap='gray')

    for (x, y) in expobj.target_coords_all:
        plt.scatter(x=x, y=y, edgecolors='yellowgreen', facecolors='none', linewidths=1.0)

    plt.suptitle('SLM targets location')

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            plt.show()
    else:
        plt.show()


### plot entire trace of individual targeted cells as super clean subplots, with the same y-axis lims
def plot_photostim_traces(array, expobj, title='', y_min=None, y_max=None, x_label=None,
                          y_label=None, save_fig=None, **kwargs):
    """

    :param array:
    :param expobj:
    :param title:
    :param y_min:
    :param y_max:
    :param x_label:
    :param y_label:
    :param save_fig:
    :param kwargs:
        options include:
            hits: list; a list of 1s and 0s that is used to add a scatter point to the plot at stim_start_frames indexes at 1s
    :return:
    """
    # make rolling average for these plots
    w = 30
    array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in array]

    len_ = len(array)
    fig, axs = plt.subplots(nrows=len_, sharex=True, figsize=(20, 3 * len_))
    for i in range(len(axs)):
        axs[i].plot(array[i], linewidth=1, color='black', zorder=2)
        if y_min != None:
            axs[i].set_ylim([y_min, y_max])
        for j in expobj.stim_start_frames:
            axs[i].axvline(x=j, c='gray', alpha=0.7, zorder=1)
        if 'scatter' in kwargs.keys():
            x = expobj.stim_start_frames[kwargs['scatter'][i]]
            y = [0] * len(x)
            axs[i].scatter(x, y, c='chocolate', zorder=3)
        if len_ == len(expobj.s2p_cell_targets):
            axs[i].set_title('Cell # %s' % expobj.s2p_cell_targets[i])
        if 'line_ids' in kwargs:
            axs[i].legend(['Target %s' % kwargs['line_ids'][i]], loc='upper left')


    axs[0].set_title((title + ' - %s' % len_ + ' cells'), loc='left', verticalalignment='top', pad=20,
                     fontsize=15)
    axs[-1].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)

    if save_fig is not None:
        plt.savefig(save_fig)

    plt.show()


def plot_photostim_traces_overlap(array, expobj, exclude_id=[], spacing=1, title='', y_lims=None,
                                  x_label='Time (seconds)', save_fig=None):
    '''
    :param array:
    :param expobj:
    :param spacing: a multiplication factor that will be used when setting the spacing between each trace in the final plot
    :param title:
    :param y_min:
    :param y_max:
    :param x_label:
    :param save_fig:
    :output: matplotlib plot
    '''
    # make rolling average for these plots
    w = 30
    array = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in array])

    len_ = len(array)
    fig, ax = plt.subplots(figsize=(40, 6))
    for i in range(len_):
        if i not in exclude_id:
            ax.plot(array[i] + i * 100 * spacing, linewidth=1)
    for j in expobj.stim_start_frames:
        if j <= array.shape[1]:
            ax.axvline(x=j, c='gray', alpha=0.3)

    ax.margins(0)
    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = int(round(item / expobj.fps))
    ax.set_xticklabels(labels)
    ax.set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(x_label)

    if y_lims is not None:
        ax.set_ylim(y_lims)

    if save_fig is not None:
        plt.savefig(save_fig)

    plt.show()


### photostim analysis - PLOT avg over all photstim. trials traces from PHOTOSTIM TARGETTED cells
def plot_periphotostim_avg(arr, expobj, stim_duration, pre_stim=10, post_stim=200, title='', y_lims=None,
                           x_label=None, y_label=None, **kwargs):
    """
    plot trace across all stims
    :param arr:
    :param expobj:
    :param stim_duration:
    :param pre_stim:
    :param post_stim:
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
    if pre_stim and post_stim:
        arr = arr[:, expobj.pre_stim - pre_stim: expobj.pre_stim + post_stim]
        x = list(range(-pre_stim, post_stim))
    else:
        x = list(range(arr.shape[1]))

    len_ = len(arr)
    flu_avg = np.mean(arr, axis=0)

    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = [5, 5]
    fig, ax = plt.subplots(figsize=figsize)
    ax.margins(0)
    ax.axvspan(0, stim_duration, alpha=0.2, color='tomato')
    for cell_trace in arr:
        if 'edgecolor' in kwargs.keys():
            ax.plot(x, cell_trace, linewidth=1, alpha=0.6, c=kwargs['edgecolor'], zorder=1)
        else:
            ax.plot(x, cell_trace, linewidth=1, alpha=0.5, zorder=1)
    ax.plot(x, flu_avg, color='black', linewidth=2.3, zorder=2)  # plot average trace
    ax.set_ylim(y_lims)
    ax.set_title((title + ' - %s' % len_ + ' traces'), horizontalalignment='center', verticalalignment='top', pad=60,
                 fontsize=10, wrap=True)

    # change x axis ticks to seconds
    if 'time' in x_label or 'Time' in x_label:
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = round(item / expobj.fps, 2)
        ax.set_xticklabels([label_format.format(x) for x in labels])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if 'savepath' in kwargs.keys():
        plt.savefig(kwargs['savepath'])

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            plt.show()
    else:
        plt.show()


def plot_s2p_raw(expobj, cell_id):
    plt.figure(figsize=(50, 3))
    plt.plot(expobj.baseline_raw[expobj.cell_id.index(cell_id)], linewidth=0.5, c='black')
    plt.xlim(0, len(expobj.baseline_raw[0]))
    plt.show()


### (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell
def plot_flu_trace(expobj, cell, x_lims=None, slm_group=None, to_plot='raw', figsize=(20, 3), linewidth=0.10, show=True):
    idx = expobj.cell_id.index(cell)
    raw = expobj.raw[idx]
    raw_ = np.delete(raw, expobj.photostim_frames)  # this is very problematic for the dFF plotting with stim frames if you're deleting ALL of the photostim frames!?!!!
    raw_dff = aoutils.normalize_dff(raw_)
    std_dff = np.std(raw_dff, axis=0)
    std = np.std(raw_, axis=0)

    # find locations along time when the trace rises above 2.5std of the mean
    x = []
    # y = []
    for j in np.arange(len(raw_dff), step=4):
        avg = np.mean(raw_dff[j:j + 4])
        if avg > np.mean(raw_dff) + 2 * std_dff:
            x.append(j)
            # y.append(0)

    if to_plot == 'raw':
        to_plot_ = raw
        to_thresh = std
    elif to_plot == 'dff':
        to_plot_ = raw_dff
        to_thresh = std_dff
    else:
        ValueError('specify to_plot as either "raw" or "dff"')

    # make the plot either as just the raw trace or as a dFF trace with the std threshold drawn as well.
    plt.figure(figsize=figsize)
    plt.plot(to_plot_, linewidth=linewidth)
    if to_plot == 'raw':
        plt.suptitle(('raw flu for cell #%s' % expobj.cell_id[idx]), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=1.00)
    elif to_plot == 'dff':
        plt.scatter(x, y=[0] * len(x), c='r', linewidth=0.1)
        plt.axhline(y=np.mean(to_plot_) + 2.5 * to_thresh, c='green')
        plt.suptitle(('dff flu for cell #%s' % expobj.cell_id[idx]), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=0.95)

    if slm_group is not None:
        for i in expobj.stim_start_frames[slm_group::expobj.n_groups]:  # select SLM group specific stim trigger frames (may not exist by each individual SLM group though...)
            plt.axvline(x=i - 1, c='gray', alpha=0.1)
    else:
        for i in expobj.stim_start_frames:  # select all stim trigger frames from the trial
            plt.axvline(x=i - 1, c='gray', alpha=0.1)

    if len(expobj.seizure_frames) > 0:
        plt.scatter(expobj.seizure_frames, y=[-20] * len(x), c='g', linewidth=0.10)


    if x_lims:
        plt.xlim(x_lims)

    # plt.ylim(0, 300)
    if show:
        plt.show()


# make a plot with the paq file LFP signal to visualize these classifications
def plot_lfp_stims(expobj, title='LFP signal with photostim. shown (in different colors relative to seizure timing',
                   x_axis: str = 'paq', sz_markings: bool = True, **kwargs):
    if 'figsize' in kwargs.keys():
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize=[20, 3])

    # plot LFP signal
    # ax.plot(expobj.lfp_signal, zorder=0, linewidth=0.5)
    fig, ax = plotLfpSignal(expobj, fig=fig, ax=ax, stim_lines=False, show=False, stim_span_color='', x_axis=x_axis, sz_markings=sz_markings)
    # y_loc = np.percentile(expobj.lfp_signal, 75)
    y_loc = 0


    # collect and plot stim times (with coloring according to sz times if available)
    # note that there is a crop adjustment to the paq-times which is needed to sync up the stim times with the plot being returned from plotLfpSignal (which also on its own crops the LFP signal)
    if 'post' in expobj.metainfo['exptype'] and '4ap' in expobj.metainfo['exptype'] and hasattr(expobj, 'stims_in_sz'):
        ax2 = ax.twinx()
        x = [(expobj.stim_start_times[expobj.stim_start_frames.index(stim)] - expobj.frame_start_time_actual) for stim in expobj.stims_in_sz]
        x_out = [(expobj.stim_start_times[expobj.stim_start_frames.index(stim)] - expobj.frame_start_time_actual) for stim in expobj.stims_out_sz
                 if stim not in expobj.stims_bf_sz and stim not in expobj.stims_af_sz]
        x_bf = [(expobj.stim_start_times[expobj.stim_start_frames.index(stim)] - expobj.frame_start_time_actual) for stim in expobj.stims_bf_sz]
        x_af = [(expobj.stim_start_times[expobj.stim_start_frames.index(stim)] - expobj.frame_start_time_actual) for stim in expobj.stims_af_sz]

        ax2.scatter(x=x, y=[y_loc] * len(expobj.stims_in_sz), edgecolors='red', facecolors='green', marker="|", zorder=3, s=60, linewidths=2.0)
        ax2.scatter(x=x_out, y=[y_loc] * len(x_out), edgecolors='grey', facecolors='black', marker="|", zorder=3, s=60, linewidths=2.0)
        ax2.scatter(x=x_bf, y=[y_loc] * len(expobj.stims_bf_sz), edgecolors='grey', facecolors='deeppink', marker="|", zorder=3, s=60, linewidths=2.0)
        ax2.scatter(x=x_af, y=[y_loc] * len(expobj.stims_af_sz), edgecolors='grey', facecolors='hotpink', marker="|", zorder=3, s=60, linewidths=2.0)
    else:
        ax2 = ax.twinx()
        x = [(expobj.stim_start_times[np.where(expobj.stim_start_frames == stim)[0][0]] - expobj.frame_start_time_actual) for stim in expobj.stim_start_frames]
        ax2.scatter(x=x, y=[y_loc] * len(x), edgecolors='red', facecolors='black', marker="|", zorder=3, s=60, linewidths=2.0)

    ax2.set_ylim([-0.0005, 0.1])
    ax2.yaxis.set_tick_params(right=False,
                              labelright=False)

    # # set x ticks at every 30 seconds
    # labels = list(range(0, len(expobj.lfp_signal)//expobj.paq_rate, 30))
    # plt.xticks(ticks=[(label * expobj.paq_rate) for label in labels], labels=labels)
    # ax.tick_params(axis='both', which='both', length=3)
    # ax.set_xlabel('Time (secs)')

    # ax.set_xticks([(label * expobj.paq_rate) for label in labels])#, labels=range(0, len(expobj.lfp_signal)//expobj.paq_rate, 30))
    # ax.set_xticklabels(labels); plt.show()

    ax.set_ylabel('LFP - voltage (mV)')

    plt.suptitle(title, wrap=True)
    plt.show()

# plot the whole pre stim to post stim period as a cool heatmap
def plot_traces_heatmap(data, vmin=None, vmax=None, stim_on=None, stim_off=None, figsize=None, title=None, xlims=(0,100),
                        cmap='bwr', show=True, **kwargs):
    """
    plot the whole pre stim to post stim period as a cool heatmap
    :param data:
    :param vmin:
    :param vmax:
    :param stim_on:
    :param stim_off:
    :param figsize:
    :param title:
    :param xlims:
    :return:
    """
    if figsize:
        fig = plt.subplots(figsize=figsize)
    else:
        fig = plt.subplots(figsize=(5, 5))
    plt.imshow(data, aspect='auto')
    plt.set_cmap(cmap)
    plt.clim(vmin, vmax)
    if xlims is not None:
        plt.xlim(xlims)
    if vmin and vmax:
        cbar = plt.colorbar(boundaries=np.linspace(vmin, vmax, 1000), ticks=[vmin, 0, vmax])
    if stim_on and stim_off:  # draw vertical dashed lines for stim period
        # plt.vlines(x=stim_on, ymin=0, ymax=len(data), colors='black')
        # plt.vlines(x=stim_off, ymin=0, ymax=len(data))
        # plt.axvline(x=stim_on, edgecolor='grey', linestyle='--')
        if type(stim_on) is int:
            stim_on = [stim_on]
            stim_off = [stim_off]
        for line in stim_on:
            plt.axvline(x=line, color='black', linestyle='--')
        for line in stim_off:
            plt.axvline(x=line, color='black', linestyle='--')
        plt.ylim(0, len(data)-0.5)

    if 'lfp_signal' in kwargs.keys():
        x_c = np.linspace(0, data.shape[1] - 1, len(kwargs['lfp_signal']))
        plt.plot(x_c, kwargs['lfp_signal'] * 50 + data.shape[0] - 100, c='black')

    plt.suptitle(title)

    if show:
        plt.show()
    else:
        pass

# plot to show the response magnitude of each cell as the actual's filling in the cell's ROI pixels.
def xyloc_responses(expobj, to_plot='dfstdf', clim=[-10, +10], plot_target_coords=True, save_fig: str = None):
    """
    plot to show the response magnitude of each cell as the actual's filling in the cell's ROI pixels.

    :param expobj:
    :param to_plot:
    :param clim:
    :param plot_target_coords: bool, if True plot the actual X and Y coords of all photostim cell targets
    :param save_fig: where to save the save figure (optional)
    :return:
    """
    stim_timings = [str(i) for i in
                    expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

    if to_plot == 'dfstdf':
        average_responses = expobj.dfstdf_all_cells[stim_timings].mean(axis=1).tolist()
    elif to_plot == 'dff':
        average_responses = expobj.dff_responses_all_cells[stim_timings].mean(axis=1).tolist()
    else:
        raise Exception('need to specify to_plot arg as either dfstdf or dff in string form!')

    # make a matrix containing pixel locations and responses at each of those pixels
    responses = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')

    for n in expobj.good_cells:
        idx = expobj.cell_id.index(n)
        ypix = expobj.stat[idx]['ypix']
        xpix = expobj.stat[idx]['xpix']
        responses[ypix, xpix] = 100. + 1 * round(average_responses[expobj.good_cells.index(n)], 2)

    # mask some 'bad' data, in your case you would have: data < 0.05
    responses = np.ma.masked_where(responses < 0.05, responses)
    cmap = plt.cm.bwr
    cmap.set_bad(color='black')

    plt.figure(figsize=(7, 7))
    im = plt.imshow(responses, cmap=cmap)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(to_plot)

    plt.clim(100+clim[0], 100+clim[1])
    if plot_target_coords:
        for (x, y) in expobj.target_coords_all:
            plt.scatter(x=x, y=y, edgecolors='green', facecolors='none', linewidths=1.0)
    plt.suptitle((expobj.metainfo['trial'] + ' - avg. %s - targets in green' % to_plot), y=0.95, fontsize=10)
    # pj.plot_cell_loc(expobj, cells=expobj.s2p_cell_targets, background_transparent=True)
    plt.show()
    if save_fig is not None:
        plt.savefig(save_fig)


# plots the raw trace for the Flu mean of the FOV (similar to the ZProject in Fiji)
def plotMeanRawFluTrace(expobj, stim_span_color='white', stim_lines: bool = True, title='raw Flu trace', x_axis='time',
                        **kwargs):
    """make plot of mean Ca trace averaged over the whole FOV"""

    # if there is a fig and ax provided in the function call then use those, otherwise start anew
    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            if 'xlims' in kwargs.keys():
                fig, ax = plt.subplots(figsize=[10 * (kwargs['xlims'][1] - kwargs['xlims'][0]) / 2000, 3])
                fig.tight_layout(pad=0)
            else:
                fig, ax = plt.subplots(figsize=[10 * len(expobj.meanRawFluTrace) / 2000, 3])
                fig.tight_layout(pad=0)

    ax.plot(expobj.meanRawFluTrace, c='forestgreen', zorder=1, linewidth=2)
    ax.margins(0)
    if stim_span_color is not None:
        if hasattr(expobj, 'shutter_frames'):
            for start, end in zip(expobj.shutter_start_frames[0], expobj.shutter_end_frames[0]):
                ax.axvspan(start-4.5, end, color=stim_span_color, zorder=2)
        else:
            for stim in expobj.stim_start_frames:
                ax.axvspan(stim-2, 1 + stim + expobj.stim_duration_frames, color=stim_span_color, zorder=2)
    if stim_lines:
        if stim_span_color is not None:
            for line in expobj.stim_start_frames:
                plt.axvline(x=line, color='black', linestyle='--', linewidth=0.6, zorder=2)
        else:
            for line in expobj.stim_start_frames:
                plt.axvline(x=line, color='black', linestyle='--', linewidth=0.6, zorder=0)
    if x_axis == 'time':
        # change x axis ticks to every 30 seconds
        labels = list(range(0, int(len(expobj.meanRawFluTrace) // expobj.fps), 30))
        ax.set_xticks(ticks=[(label * expobj.fps) for label in labels])
        # for item in labels:
        #     labels[labels.index(item)] = int(round(item / expobj.fps))
        # ticks_loc = ax.get_xticks().tolist()
        # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('frame clock')
    ax.set_ylabel('Flu (a.u.)')
    if 'xlims' in kwargs.keys():
        ax.set_xlim(kwargs['xlims'])

    # add title
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))

    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            plt.show()
        else:
            pass
    else:
        plt.show()

    if 'fig' in kwargs.keys():
        return fig, ax


# plots the raw trace for the Flu mean of the FOV
def plotLfpSignal(expobj, stim_span_color='powderblue', stim_lines: bool = True, sz_markings: bool = False, title='LFP trace', x_axis='time', **kwargs):
    """make plot of LFP with also showing stim locations"""

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
    ax.plot(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual], c=color, zorder=1, linewidth=0.4, alpha=alpha)
    ax.margins(0)

    # plot stims
    if stim_span_color != '':
        for stim in expobj.stim_start_times:
            stim = stim - expobj.frame_start_time_actual
            ax.axvspan(stim - 8, 1 + stim + expobj.stim_duration_frames / expobj.fps * expobj.paq_rate, color=stim_span_color, zorder=1, alpha=0.5)
    else:
        if stim_lines:
            for line in expobj.stim_start_times:
                line = line - expobj.frame_start_time_actual
                plt.axvline(x=line+2, color='black', linestyle='--', linewidth=0.6, zorder=0)

    # plot seizure onset and offset markings
    if sz_markings:
        if hasattr(expobj, 'seizure_lfp_onsets'):
            for sz_onset in expobj.seizure_lfp_onsets:
                plt.axvline(x=expobj.frame_clock_actual[sz_onset] - expobj.frame_start_time_actual, color='black', linestyle='--', linewidth=1.0, zorder=0)
            for sz_offset in expobj.seizure_lfp_offsets:
                plt.axvline(x=expobj.frame_clock_actual[sz_offset] - expobj.frame_start_time_actual, color='black', linestyle='--', linewidth=1.0, zorder=0)

    # change x axis ticks to seconds
    if x_axis == 'time':
        # set x ticks at every 30 seconds
        labels = list(range(0, len(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual]) // expobj.paq_rate, 30))
        ax.set_xticks(ticks=[(label * expobj.paq_rate) for label in labels])
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='both', length=3)
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
        ax.set_ylim([np.mean(expobj.lfp_signal) - 5, np.mean(expobj.lfp_signal) + 5])

    # add title
    plt.suptitle(
        '%s - %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))

    # options for showing plot or returning plot
    if 'show' in kwargs.keys():
        if kwargs['show'] is True:
            plt.show()
        else:
            pass
    else:
        plt.show()

    if 'fig' in kwargs.keys():
        return fig, ax


def plot_flu_1pstim_avg_trace(expobj, title='Average trace of stims', individual_traces=False, x_axis='time', stim_span_color='white',
                              y_axis: str = 'raw', quantify: bool = False, stims_to_analyze: list = None):
    pre_stim = 1  # seconds
    post_stim = 4  # seconds
    fig, ax = plt.subplots()

    if stims_to_analyze is None:
        stims_to_analyze = expobj.stim_start_frames
    flu_list = [expobj.meanRawFluTrace[stim - int(pre_stim * expobj.fps): stim + int(post_stim * expobj.fps)] for stim in stims_to_analyze]
    # convert to dFF normalized to pre-stim F
    if y_axis == 'dff':  # otherwise default param is raw Flu
        flu_list = [pj.dff(trace, baseline=np.mean(trace[:int(pre_stim * expobj.fps) - 2])) for trace in flu_list]

    avg_flu_trace = np.mean(flu_list, axis=0)
    ax.plot(avg_flu_trace, color='black', zorder=2, linewidth=2.2)
    ax.margins(0)

    if individual_traces:
        # individual traces
        for trace in flu_list:
            ax.plot(trace, color='forestgreen', zorder=1, alpha=0.25)
        if stim_span_color is not None:
            ax.axvspan(int(pre_stim * expobj.fps) - 2, int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 1, color='skyblue', zorder=1, alpha=0.7)
        elif stim_span_color is None:
            plt.axvline(x=int(pre_stim * expobj.fps) - 2, color='black', linestyle='--', linewidth=1)
            plt.axvline(x=int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 1, color='black', linestyle='--', linewidth=1)
    else:
        # plot standard deviation of the traces array as a span above and below the mean
        std_ = np.std(flu_list, axis=0)
        ax.fill_between(x=range(len(avg_flu_trace)), y1=avg_flu_trace + std_, y2=avg_flu_trace - std_, alpha=0.3, zorder=1, color='forestgreen')
        if stim_span_color is not None:
            ax.axvspan(int(pre_stim * expobj.fps) - 3, int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 1.5, color=stim_span_color, zorder=3)
        elif stim_span_color is None:
            plt.axvline(x=int(pre_stim * expobj.fps) - 2, color='black', linestyle='--', linewidth=1)
            plt.axvline(x=int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 2, color='black', linestyle='--', linewidth=1)

    if x_axis == 'time':
        # change x axis ticks to seconds
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = round(item / expobj.fps, 2)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in labels])
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('frame clock')
    ax.set_ylabel(y_axis)
    fig.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


    # quantification of the stim response (compared to prestim baseline)
    if quantify:
        response_len = 0.5  # post-stim response period in sec
        poststim_1 = int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 2
        poststim_2 = poststim_1 + int(response_len * expobj.fps)
        baseline = int(pre_stim * expobj.fps) - 2
        ax.axvspan(poststim_1, poststim_2,
                   color='#ffd700', zorder=1, alpha=0.35)
        ax.axvspan(0, baseline,
                   color='#5e5d5d', zorder=1, alpha=0.15)
        response = np.mean(avg_flu_trace[poststim_1:poststim_2]) - np.mean(avg_flu_trace[:baseline])
        print('Average response %s: %s' % (y_axis, '{:,.4f}'.format(response)))

        # add the response value to the top right of the plot
        ax.text(0.98, 0.97, 'Average response %s: %s' % (y_axis, '{:,.4f}'.format(response)),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, fontweight='bold',
                color='green', fontsize=10)
        ax.text(0.015, 0.97, 'pre-stim',
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontweight='bold',
                color='#5e5d5d', fontsize=10)
        ax.text(0.265, 0.97, 'post.',
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontweight='bold',
                color='#d1ae00', fontsize=10)

    ax.set_ylim([-0.5, 1.0])
    plt.show()
    return flu_list, round(response, 4)

def plot_lfp_1pstim_avg_trace(expobj, title='Average LFP peri- stims', individual_traces=False, x_axis='time', pre_stim=1.0, post_stim=5.0,
                              optoloopback: bool = False, stims_to_analyze: list = None):
    stim_duration = int(np.mean([expobj.stim_end_times[idx] - expobj.stim_start_times[idx] for idx in range(len(expobj.stim_start_times))]) + 0.01*expobj.paq_rate)
    pre_stim = pre_stim  # seconds
    post_stim = post_stim  # seconds

    fig, ax = plt.subplots()

    if stims_to_analyze is None:
        stims_to_analyze = expobj.stim_start_times
    x = [expobj.lfp_signal[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim in stims_to_analyze]
    x_ = np.mean(x, axis=0)
    ax.plot(x_, color='black', zorder=3, linewidth=1.75)

    ax.set_ylim([np.mean(x_) - 2.5, np.mean(x_) + 2.5])
    ax.margins(0)

    if individual_traces:
        # individual traces
        for trace in x:
            ax.plot(trace, color='steelblue', zorder=1, alpha=0.25)
            # ax.axvspan(int(pre_stim * expobj.paq_rate),
            #            int(pre_stim * expobj.paq_rate) + stim_duration,
            #            edgecolor='powderblue', zorder=1, alpha=0.3)
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration,
                   color='skyblue', zorder=1, alpha=0.7)

    else:
        # plot standard deviation of the traces array as a span above and below the mean
        std_ = np.std(x, axis=0)
        ax.fill_between(x=range(len(x_)), y1=x_ + std_, y2=x_ - std_, alpha=0.3, zorder=2, color='steelblue')
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration, color='skyblue', zorder=1, alpha=0.7)


    if optoloopback:
        ax2 = ax.twinx()
        if not hasattr(expobj, 'opto_loopback'):
            print('loading', expobj.paq_path)

            paq, _ = paq_read(expobj.paq_path, plot=False)
            expobj.paq_rate = paq['rate']

            # find voltage channel and save as lfp_signal attribute
            voltage_idx = paq['chan_names'].index('opto_loopback')
            expobj.opto_loopback = paq['data'][voltage_idx]
            expobj.save()
        else:
            pass
        x = [expobj.opto_loopback[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim
             in expobj.stim_start_times]
        y_avg = np.mean(x, axis=0)
        ax2.plot(y_avg, color='royalblue', zorder=3, linewidth=1.75)
        ax2.text(0.98, 0.12, 'Widefield LED TTL',
                 transform=ax.transAxes, fontweight='bold', horizontalalignment='right',
                 color='royalblue', fontsize=10)
        # ax2.set_ylabel('Widefield LED TTL', color='royalblue', fontweight='bold')
        ax2.yaxis.set_tick_params(right=False,
                                  labelright=False)
        ax2.set_ylim([-3, 30])
        ax2.margins(0)



    if x_axis == 'time':
        # change x axis ticks to seconds
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = round(item / expobj.paq_rate, 2)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in labels])
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('paq clock')
    ax.set_ylabel('Voltage')

    # add title
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

### below are plotting functions that I am still working on coding:

