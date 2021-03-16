# library of functions that are used for making various plots for all optical photostimulation/imaging experiments
# the data that feeds into these plots are generated by Vape (from alloptical_analysis_photostim.py)

# imports
import numpy as np
import matplotlib.pyplot as plt

import alloptical_utils_pj as aoutils


### plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????
def plot_cell_radius_aspectr(expobj, stat, to_plot):
    radius = []
    aspect_ratio = []
    for cell in range(len(stat)):
        # if expobj.cell_id[cell] in expobj.good_cells:
        if expobj.cell_id[cell] in expobj.cell_id:
            radius.append(stat[cell]['radius'])
            aspect_ratio.append(stat[cell]['aspect_ratio'])

    if to_plot == 'radius':
        to_plot_ = radius
    elif to_plot == 'aspect':
        to_plot_ = aspect_ratio
    n, bins, patches = plt.hist(to_plot_, 100)
    plt.axvline(3.5, color='green')
    plt.axvline(8.5, color='red')
    plt.suptitle('All cells - %s' % to_plot, y=0.95)
    plt.show()
    return to_plot_


### plot entire trace of individual targeted cells as super clean subplots, with the same y-axis lims
def plot_photostim_subplots(dff_array, expobj, title='', y_min=None, y_max=None, x_label=None,
                            y_label=None, save_fig=None):
    # make rolling average for these plots
    w = 30
    dff_array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array]

    len_ = len(dff_array)
    fig, axs = plt.subplots(nrows=len_, sharex=True, figsize=(20, 3 * len_))
    for i in range(len(axs)):
        axs[i].plot(dff_array[i], linewidth=1, color='black')
        if y_min != None:
            axs[i].set_ylim([y_min, y_max])
        for j in expobj.stim_start_frames:
            axs[i].axvline(x=j, c='gray', alpha=0.7)

    axs[0].set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                     fontsize=15)
    axs[-1].set_xlabel(x_label)
    axs[-1].set_ylabel(y_label)

    if save_fig is not None:
        plt.savefig(save_fig)

    plt.show()


def plot_photostim_overlap_plots(dff_array, expobj, spacing=1, title='', y_min=None,
                                 y_max=None, x_label='Time (seconds)', save_fig=None):
    '''
    :param dff_array:
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
    dff_array = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array])

    len_ = len(dff_array)
    fig, ax = plt.subplots(figsize=(40, 6))
    for i in range(len_):
        ax.plot(dff_array[i] + i * 100 * spacing, linewidth=1)
    for j in expobj.stim_start_frames:
        if j <= dff_array.shape[1]:
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

    if save_fig is not None:
        plt.savefig(save_fig)

    plt.show()


### photostim analysis - PLOT avg over all photstim. trials traces from PHOTOSTIM TARGETTED cells
def plot_photostim_avg(dff_array, expobj, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
                       x_label=None, y_label=None, savepath=None):
    dff_array = dff_array[:, :pre_stim + post_stim]
    len_ = len(dff_array)
    flu_avg = np.mean(dff_array, axis=0)
    x = list(range(-pre_stim, post_stim))

    fig, ax = plt.subplots()
    ax.margins(0)
    ax.axvspan(0, stim_duration, alpha=0.2, color='crimson')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth=1, alpha=0.8)
    ax.plot(x, flu_avg, color='black', linewidth=2)  # plot median trace
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    ax.set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = round(item / expobj.fps, 1)
    ax.set_xticklabels(labels)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    plt.show()


### (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell
def plot_flu_trace(expobj, cell, x_lims=None, slm_group=None, to_plot='raw', figsize=(20, 3), linewidth=0.10):
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
        if avg > np.mean(raw_dff) + 2.5 * std_dff:
            x.append(j)
            # y.append(0)

    print(x)

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



        if title is not None:
            plt.suptitle(title)
        plt.show()
    else:
        raise Exception('look, you need to create stims_in_sz and stims_out_sz attributes first (or rewrite this function)')

def plot_heatmap_photostim_trace(data, vmin=None, vmax=None, stim_on=None, stim_off=None, figsize=None):
    if figsize:
        fig = plt.subplots(figsize=figsize)
    else:
        fig = plt.subplots(figsize=(5, 5))
    plt.imshow(data, aspect='auto')
    plt.set_cmap('bwr')
    plt.clim(vmin, vmax)
    plt.xlim(0, 100)
    if vmin and vmax:
        cbar = plt.colorbar(boundaries=np.linspace(vmin, vmax, 1000), ticks=[vmin, 0, vmax])
    if stim_on and stim_off: # draw vertical dashed lines for stim period
        # plt.vlines(x=stim_on, ymin=0, ymax=len(data), colors='black')
        # plt.vlines(x=stim_off, ymin=0, ymax=len(data))
        plt.axvline(x=stim_on, color='black', linestyle='--')
        plt.axvline(x=stim_off, color='black', linestyle='--')
        plt.ylim(0, len(data)-0.5)
    plt.show()

### below are plotting functions that I am still working on coding:
