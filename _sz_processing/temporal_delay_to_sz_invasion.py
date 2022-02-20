import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from pathlib import Path

import pandas as pd

import _alloptical_utils as Utils

from _main_.Post4apMain import Post4ap

SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"

# expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

# expobj.slmtargets_data.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]

# %%

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=True, ignore_cache=True)
def _export_data_(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    from _utils_._lfp import LFP
    lfp, downlfp = LFP.downsampled_LFP(expobj)

    lfp_indexes = np.linspace(0, len(lfp) - 1, expobj.n_frames)
    lfp_indexes = [int(x) for x in lfp_indexes]

    lfp_fr = lfp[lfp_indexes]
    num = expobj.raw_SLMTargets.shape[0] + 3

    sz_marks = np.array([False] * expobj.n_frames)
    for i in range(expobj.n_frames):
        if i in expobj.seizure_lfp_onsets or i in expobj.seizure_lfp_offsets:
            sz_marks[i] = True

    # build array for export
    array = np.empty(shape=(num, expobj.n_frames))

    array[0] = lfp_fr
    array[1] = sz_marks
    array[2] = expobj.meanRawFluTrace

    for i in range(expobj.raw_SLMTargets.shape[0]):
        arr = expobj.raw_SLMTargets[i]
        array[i + 3] = arr

    path = expobj.analysis_save_path + '/export/'
    os.makedirs(path, exist_ok=True)
    path_flu = path + f"{expobj.prep}_{expobj.trial}_slmtargets_array.npy"
    # path_lfp = path + f"{expobj.prep}_{expobj.trial}_lfp_array.npy"
    print(f"saving to {path_flu}")
    # print(path_lfp)
    print('\n')

    with open(path_flu, 'wb') as f:
        np.save(path_flu, array)

    # with open(path, 'wb') as f:
    #     np.save(f, lfp)


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, supress_print=True, ignore_cache=True)
def print_pkl(result, **kwargs):
    expobj: Post4ap = kwargs['expobj']
    print(expobj.pkl_path)
    # result.append(expobj.pkl_path)
    return result



def plot_slmtargets_flu_lfp(expobj: Post4ap, rg: Tuple[int] = None, title='', **kwargs):
    nparray = expobj.raw_SLMTargets[rg[0]:rg[1]] if rg else expobj.raw_SLMTargets

    len_ = nparray.shape[0]
    print(f"len={len_}")
    # make rolling average for these plots
    w = 30
    array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in nparray]
    print(f"len of array: {len(array)}")

    figsize = kwargs['figsize'] if 'figsize' in [*kwargs] else (24, 5 * (len_ // 6))
    fig, axs = plt.subplots(figsize=figsize)
    # fig, axs = plt.subplots(figsize=(24,5))

    # ax2 = axs.twinx()

    # LFP
    from _utils_._lfp import LFP
    lfp, downlfp = LFP.downsampled_LFP(expobj)

    # lfp_indexes = range(0, len(lfp), len(lfp)//expobj.n_frames)
    lfp_indexes = np.linspace(0, len(lfp) - 1, nparray.shape[1])
    lfp_indexes = [int(x) for x in lfp_indexes]

    lfp_fr = lfp[lfp_indexes] * 100 + 800

    axs.plot(lfp_fr, color='blue', lw=0.5)  # LFP trace
    # axs.set_ylim(np.mean(lfp_fr) - 10, np.mean(lfp_fr) + 45)
    # seizure markings - checkig also if there is a wavefront that reaches FOV (dont analyse seizures without that)
    for i, j in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
        stims = [stim for stim in expobj.stimsWithSzWavefront if i < stim < j]
        if len(stims) > 0:
            axs.axvline(i, color='lightcoral', zorder=0)
            axs.axvline(j, color='skyblue', zorder=0)
    # sz_marks = np.where(nparray[1] == 1)[0]
    # print(sz_marks)
    # for i in sz_marks:
    #     ax2.axvline(i)

    axs.plot((np.convolve(expobj.meanRawFluTrace, np.ones(w), 'valid') / w)-1000, color='green')  # mean raw flu of FOV
    for i in range(len_):
        arr_ = array[i] + ((i+1) * 3000)
        axs.plot(arr_, color='dimgrey', alpha=0.75)  # cell flu trace
        arr2 = [arr_[i] for i in range(len(arr_)) if i not in expobj.seizure_frames]
        axs.axhline(np.mean(arr2)+np.std(arr2), color='khaki', zorder = 0, alpha=0.2)
        axs.axhline(np.mean(arr2)-np.std(arr2), color='khaki', zorder = 0, alpha=0.2)

    axs.margins(0)
    axs.margins(0)
    # fig.tight_layout(pad=0.5)
    axs.set_title(title)
    fig.show()

    # k = Klickers([axs])

    return axs

def evaluate_clicks_sz(expobj: Post4ap, klicker, slmtargets_time_delay_sz: pd.DataFrame = None):
    "note that this code is run directly on local machine as part of local analysis pipeline"
    try:
        fps = expobj.fps
    except FileNotFoundError:
        if 'RL' in expobj.prep:
            fps = 30
        else:
            fps=15

    if slmtargets_time_delay_sz is None:
        _columns = [f"time_del_sz{num}" for num in range(expobj.numSeizures)]
        df = pd.DataFrame(index=range(expobj.n_targets_total), columns=_columns)
    else:
        df = slmtargets_time_delay_sz

    for target in [*klicker.get_positions()]:
        if len(klicker.get_positions()[target]) > 0:
            slm_target = target[6:]
            _recruitment_stamps = klicker.get_positions()[target][:,0]
            for start, stop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
                sz_num = expobj.seizure_lfp_onsets.index(start)
                stamps = [click for click in _recruitment_stamps if start < click < stop]
                if len(stamps) > 0:
                    prime_point = np.min(stamps)
                    time_diff = round(((prime_point - start) / fps), 3)  # time difference (secs) between start and first sz point (if multiple)
                    col = f"time_del_sz{sz_num}"
                    df.loc[int(slm_target), col] = time_diff

    return df

def export_to_csv(slmtargets_time_delay_sz: pd.DataFrame, exp_name: str):
    filepath = Path(SAVE_LOC + exp_name + '.csv')
    Utils.save_to_csv(slmtargets_time_delay_sz, savepath=filepath)


# ADD SLM TARGETS TIME DELAY TO SZ DATA TO expobj

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
def add_slmtargets_time_delay_sz_data(**kwargs):
    expobj = kwargs['expobj']
    csv_path = f'/home/pshah/mnt/qnap/Analysis/analysis_export/slmtargets_time_delay_sz__{expobj.prep}_{expobj.trial}.csv'
    slmtargets_time_delay_sz = pd.read_csv(csv_path)
    expobj.slmtargets_time_delay_sz = slmtargets_time_delay_sz
    print(f"adding slmtargets time delay to {expobj.t_series_name}")
    for column in slmtargets_time_delay_sz.columns[1:]:
        expobj.slmtargets_data.add_observation(obs_name=column, values=slmtargets_time_delay_sz[column])
    expobj.save()


def convert_timedel2frames(expobj: Post4ap, sz_num: int, timestamp: float):

    start, stop = expobj.seizure_lfp_onsets[sz_num], expobj.seizure_lfp_offsets[sz_num]
    numFrames = timestamp * expobj.fps
    frameNumber = numFrames + start

    return int(round(frameNumber))


### ARCHIVE



# def plot_klickers_photostim_traces(array, expobj: Post4ap, title='', y_min=None, y_max=None, x_label=None,
#                                    y_label=None, save_fig=None, **kwargs):
#     """
#
#     :param array:
#     :param expobj:
#     :param title:
#     :param y_min:
#     :param y_max:
#     :param x_label:
#     :param y_label:
#     :param save_fig:
#     :param kwargs:
#         options include:
#             hits: ls; a ls of 1s and 0s that is used to add a scatter point to the plot at stim_start_frames indexes at 1s
#     :return:
#     """
#     # make rolling average for these plots
#     w = 30
#     array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in array]
#
#     len_ = len(array) + 2
#     fig, axs = plt.subplots(nrows=len_, figsize=(100, 3 * len_))
#
#     aoplot.plotMeanRawFluTrace(expobj=expobj, fig = fig, ax=axs[0], show=False)
#     aoplot.plot_lfp_stims(expobj=expobj, fig = fig, ax=axs[1], show=False)
#
#     for i, ax in enumerate(axs[2:]):
#         ax.plot(array[i], linewidth=1, color='black', zorder=2)
#         ax.margins(0.02)
#         if y_min != None:
#             ax.set_ylim([y_min, y_max])
#         for j in expobj.stim_start_frames:
#             ax.axvline(x=j, c='gray', alpha=0.7, zorder=1)
#         if 'scatter' in kwargs.keys():
#             x = expobj.stim_start_frames[kwargs['scatter'][i]]
#             y = [0] * len(x)
#             ax.scatter(x, y, c='chocolate', zorder=3)
#         if len_ == len(expobj.s2p_cell_targets):
#             ax.set_title('Cell # %s' % expobj.s2p_cell_targets[i])
#         if 'line_ids' in kwargs:
#             ax.legend(['Target %s' % kwargs['line_ids'][i]], loc='upper left')
#
#         # change x axis ticks to every 5 seconds
#         labels = list(range(0, int(expobj.n_frames // expobj.fps), 5))
#         ax.set_xticks(ticks=[(label * expobj.fps) for label in labels])
#         ax.set_xticklabels(labels)
#         ax.set_xlabel('Time (secs)')
#
#
#     axs[0].set_title((title + ' - %s' % len_ + ' cells'), loc='left', verticalalignment='top', pad=20,
#                      fontsize=15)
#     # axs[0].set_xlabel(x_label)
#     # axs[0].set_ylabel(y_label)
#
#     fig.tight_layout(pad=0.02)
#
#     # fig.show()
#     Utils.save_figure(fig, save_path_suffix=f"SLM_targets_{expobj.prep}_{expobj.trial}.png")
#
#
#     # klickers = aoplot.Klickers(axs=axs)
#     # return klickers
#
# plot_klickers_photostim_traces(array=expobj.raw_SLMTargets[:2], expobj=expobj)
