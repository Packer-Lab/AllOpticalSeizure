# library of functions that are used for making various plots for all optical photostimulation/imaging experiments
# the data that feeds into these plots are generated by Vape (from alloptical_analysis_photostim.py)

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import alloptical_utils_pj as aoutils
import tifffile as tf


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
            'color': str, color of the individual traces behind the mean trace
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

    fig, ax = plt.subplots()
    ax.margins(0)
    ax.axvspan(0, stim_duration, alpha=0.2, color='tomato')
    for cell_trace in arr:
        if 'color' in kwargs.keys():
            ax.plot(x, cell_trace, linewidth=1, alpha=0.6, c=kwargs['color'], zorder=1)
        else:
            ax.plot(x, cell_trace, linewidth=1, alpha=0.5, zorder=1)
    ax.plot(x, flu_avg, color='black', linewidth=2.3, zorder=2)  # plot average trace
    ax.set_ylim(y_lims)
    ax.set_title((title + ' - %s' % len_ + ' traces'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # change x axis ticks to seconds
    if x_label == 'time':
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = round(item / expobj.fps, 2)
        ax.set_xticklabels([label_format.format(x) for x in labels])

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
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
def plot_lfp_stims(expobj, title=None):
    if hasattr(expobj, 'stims_in_sz') and hasattr(expobj, 'stims_out_sz'):
        fig, ax = plt.subplots(figsize=[20, 3])
        x = [expobj.stim_times[np.where(expobj.stim_start_frames == stim)[0][0]] for stim in expobj.stims_in_sz]
        x_out = [expobj.stim_times[np.where(expobj.stim_start_frames == stim)[0][0]] for stim in expobj.stims_out_sz
                 if stim not in expobj.stims_bf_sz and stim not in expobj.stims_af_sz]
        x_bf = [expobj.stim_times[np.where(expobj.stim_start_frames == stim)[0][0]] for stim in expobj.stims_bf_sz]
        x_af = [expobj.stim_times[np.where(expobj.stim_start_frames == stim)[0][0]] for stim in expobj.stims_af_sz]
        ax.plot(expobj.lfp_signal)
        ax.scatter(x=x, y=[0] * len(expobj.stims_in_sz), edgecolors='red', facecolors='white')
        ax.scatter(x=x_out, y=[0] * len(x_out), edgecolors='grey', facecolors='white')
        ax.scatter(x=x_bf, y=[0] * len(expobj.stims_bf_sz), edgecolors='grey', facecolors='deeppink')
        ax.scatter(x=x_af, y=[0] * len(expobj.stims_af_sz), edgecolors='grey', facecolors='hotpink')
        # set x ticks at every 30 seconds
        labels = list(range(0, len(expobj.lfp_signal)//expobj.paq_rate, 30))
        plt.xticks(ticks=[(label * expobj.paq_rate) for label in labels], labels=labels)
        ax.tick_params(axis='both', which='both', length=3)
        # ax.set_xticks([(label * expobj.paq_rate) for label in labels])#, labels=range(0, len(expobj.lfp_signal)//expobj.paq_rate, 30))
        # ax.set_xticklabels(labels); plt.show()
        #
        ax.set_xlabel('Time (secs)')
        ax.set_ylabel('LFP - voltage (mV)')



        plt.suptitle(title)
        plt.show()
    else:
        raise Exception('look, you need to create stims_in_sz and stims_out_sz attributes first (or rewrite this function)')

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
        # plt.axvline(x=stim_on, color='grey', linestyle='--')
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


def plot_flu_trace_1pstim(expobj, stim_span_color='white', title='raw Flu trace', x_axis='time', figsize=None, xlims=None):
    # make plot of avg Ca trace
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if xlims:
            fig, ax = plt.subplots(figsize=[10 * (xlims[1] - xlims[0]) / 2000, 3])
        else:
            fig, ax = plt.subplots(figsize=[10 * len(expobj.onePstim_trace) / 2000, 3])
    ax.plot(expobj.onePstim_trace, c='forestgreen', zorder=1, linewidth=2)
    if stim_span_color is not None:
        if hasattr(expobj, 'shutter_frames'):
            for start, end in zip(expobj.shutter_start_frames[0], expobj.shutter_end_frames[0]):
                ax.axvspan(start-4.5, end, color=stim_span_color, zorder=2)
        else:
            for stim in expobj.stim_start_frames:
                ax.axvspan(stim, 1 + stim + expobj.stim_duration_frames, color=stim_span_color, zorder=2)
    if stim_span_color != 'black':
        for line in expobj.stim_start_frames:
            plt.axvline(x=line, color='black', linestyle='--', linewidth=0.6)
    if x_axis == 'time':
        # change x axis ticks to seconds
        label_format = '{:,.0f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = int(round(item / expobj.fps))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in labels])
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('frame clock')
    ax.set_ylabel('Flu (a.u.)')
    if xlims:
        ax.set_xlim(xlims)
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

def plot_1pstim_avg_trace(expobj, title='Average trace of stims', individual_traces=False, x_axis='time', stim_span_color='white'):
    pre_stim = 1  # seconds
    post_stim = 4  # seconds
    fig, ax = plt.subplots()
    x = [expobj.onePstim_trace[stim - int(pre_stim * expobj.fps): stim + int(post_stim * expobj.fps)] for stim in expobj.stim_start_frames]
    x_ = np.mean(x, axis=0)
    ax.plot(x_, color='black', zorder=2, linewidth=2.2)

    if individual_traces:
        # individual traces
        for trace in x:
            ax.plot(trace, color='forestgreen', zorder=1, alpha=0.25)
        if stim_span_color is not None:
            ax.axvspan(int(pre_stim * expobj.fps) - 2, int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 1, color='skyblue', zorder=1, alpha=0.7)
        elif stim_span_color is None:
            plt.axvline(x=int(pre_stim * expobj.fps) - 2, color='black', linestyle='--', linewidth=1)
            plt.axvline(x=int(pre_stim * expobj.fps) + expobj.stim_duration_frames + 1, color='black', linestyle='--', linewidth=1)
    else:
        # plot standard deviation of the traces array as a span above and below the mean
        std_ = np.std(x, axis=0)
        ax.fill_between(x=range(len(x_)), y1=x_ + std_, y2=x_ - std_, alpha=0.3, zorder=1, color='forestgreen')
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
    ax.set_ylabel('Flu (a.u.)')
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

def plot_lfp_1pstim_avg_trace(expobj, title='Average LFP peri- stims', individual_traces=False, x_axis='time', pre_stim=1.0, post_stim=5.0):
    stim_duration = int(np.mean([expobj.stim_end_times[idx] - expobj.stim_start_times[idx] for idx in range(len(expobj.stim_start_times))]) + 0.01*expobj.paq_rate)
    pre_stim = pre_stim  # seconds
    post_stim = post_stim  # seconds
    fig, ax = plt.subplots()
    x = [expobj.lfp_signal[stim - int(pre_stim * expobj.paq_rate): stim + int(post_stim * expobj.paq_rate)] for stim in expobj.stim_start_times]
    x_ = np.mean(x, axis=0)
    ax.plot(x_, color='black', zorder=3, linewidth=1.75)

    if individual_traces:
        # individual traces
        for trace in x:
            ax.plot(trace, color='steelblue', zorder=1, alpha=0.25)
            # ax.axvspan(int(pre_stim * expobj.paq_rate),
            #            int(pre_stim * expobj.paq_rate) + stim_duration,
            #            color='powderblue', zorder=1, alpha=0.3)
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration,
                   color='skyblue', zorder=1, alpha=0.7)

    else:
        # plot standard deviation of the traces array as a span above and below the mean
        std_ = np.std(x, axis=0)
        ax.fill_between(x=range(len(x_)), y1=x_ + std_, y2=x_ - std_, alpha=0.3, zorder=2, color='steelblue')
        ax.axvspan(int(pre_stim * expobj.paq_rate),
                   int(pre_stim * expobj.paq_rate) + stim_duration, color='skyblue', zorder=1, alpha=0.7)

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
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

def plot_lfp_1pstim(expobj, stim_span_color='powderblue', title='LFP trace', x_axis='time', figsize=None):
    # make plot of avg Ca trace
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=[60 * (expobj.stim_start_times[-1] + 1e5 - (expobj.stim_start_times[0] - 1e5)) / 1e7, 3])
    ax.plot(expobj.lfp_signal, c='steelblue', zorder=1, linewidth=0.4)
    if stim_span_color is not None:
        for stim in expobj.stim_start_times:
            ax.axvspan(stim - 8, 1 + stim + expobj.stim_duration_frames / expobj.fps * expobj.paq_rate, color=stim_span_color, zorder=1, alpha=0.5)
    else:
        for line in expobj.stim_start_times:
            plt.axvline(x=line+2, color='black', linestyle='--', linewidth=0.6)
    if x_axis == 'time':
        # change x axis ticks to seconds
        label_format = '{:,.2f}'
        labels = [item for item in ax.get_xticks()]
        for item in labels:
            labels[labels.index(item)] = int(round(item / expobj.paq_rate, 2))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(x) for x in labels])
        ax.set_xlabel('Time (secs)')
    else:
        ax.set_xlabel('paq clock')
    ax.set_ylabel('Voltage')
    ax.set_xlim([expobj.stim_start_times[0] - 1e5, expobj.stim_start_times[-1] + 1e5])
    plt.suptitle(
        '%s %s %s %s' % (title, expobj.metainfo['exptype'], expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
    plt.show()

### below are plotting functions that I am still working on coding:
