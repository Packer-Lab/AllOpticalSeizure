"""
quick caption:

A) representative example of post-seizure termination mean fluorescence level goes below interictal periods
B) quantification of mean FOV fluorescence in 30 secs post-seizure termination

TODO
[ ] - lfp traces timed to photostimulation timing
    - some experiments do have afterdischarges, but the key is just to make sure that there's not many afterdischarges that occuring right at the photostimulation trial time
[x] - calculate timescale of recovery to baseline
[x]- add traces of LFP -- trying to check if there's after discharges that might be causing the seeming hyper-excitable responses of neurons at the time of photostimulation timings.

"""


# %%
import sys
from typing import Union

from funcsforprajay.plotting.plotting import plot_bar_with_points
from funcsforprajay.funcs import decay_constant_logfit_method, convert_to_positive, decay_timescale
from scipy import signal
from scipy.stats import sem

from _alloptical_utils import run_for_loop_across_exps
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureResults
from _exp_metainfo_.exp_metainfo import ExpMetainfo
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings, inspectExperimentMeanFOVandLFP
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import rep_fig_vis as rfv

plot_settings()

Results: ExpSeizureResults = ExpSeizureResults.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

save_fig = True



# %% MAKE FIGURE LAYOUT
rfv.set_fontsize(10)

layout = {
    # 'A': {'panel_shape': (1, 1, 'twinx'),
    #       'bound': (0.15, 0.75, 0.45, 0.90)},
    'A': {'panel_shape': (1, 2),
          'bound': (0.15, 0.60, 0.45, 0.90),
          'hspace': 0},
    'B': {'panel_shape': (1, 1),
          'bound': (0.65, 0.75, 0.72, 0.90)}
}

dpi = 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)




# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.



# %% A - example of all seizures from RL108 t-013

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')

mean_interictal = np.mean([expobj.meanRawFluTrace[x] for x in range(expobj.n_frames) if (x not in expobj.seizure_frames and x not in expobj.photostim_frames and x < expobj.photostim_frames[-1])])
lfp_trace = expobj.lfp_signal[expobj.frame_clock][:expobj.n_frames]


lfp_traces = []
traces = []
num_frames = []
for onset, offset in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
    frames = np.arange(int(offset - expobj.fps * 10), int(offset + expobj.fps * 40))
    frames = [frame for frame in frames if frame not in expobj.photostim_frames]
    try:
        if onset > 0 and offset + int(expobj.fps * 30) < expobj.n_frames:
            pre_onset_frames = np.arange(int(onset - expobj.fps * 10), int(onset))
            pre_onset_frames = [frame for frame in pre_onset_frames if frame not in expobj.photostim_frames]
            # pre_onset = expobj.meanRawFluTrace[onset - int(expobj.fps * 30): onset]
            pre_onset = expobj.meanRawFluTrace[pre_onset_frames]
            pre_mean = np.mean(pre_onset)

            trace = expobj.meanRawFluTrace[frames]
            lfp_traces_ = lfp_trace[frames]

            # trace = expobj.meanRawFluTrace[int(offset - expobj.fps * 10): int(offset + expobj.fps * 30)]
            trace_norm = (trace / mean_interictal) * 100
            # trace_norm = (trace / pre_mean) * 100
            # ax.plot(trace_norm, lw=0.3)

            lfp_traces.append(lfp_traces_)
            traces.append(trace_norm)
            num_frames.append(len(frames))
    except:
        pass

x_range = np.linspace(0, 30, min(num_frames))
mean_ = np.mean([trace[:min(num_frames)] for trace in traces], axis=0)
sem_ = sem([trace[:min(num_frames)] for trace in traces])

# %%

# add to plot
ax = axes['A'][0]
ax2 = axes['A'][1]

# f, ax = plt.subplots(figsize=(5, 3))
ax.plot(x_range, mean_, c='black', lw=1.5)
ax.fill_between(x_range, mean_ - sem_, mean_ + sem_, alpha=0.3, color='forestgreen')
# ax.axhline(y=100, ls='--', zorder=0, color='black', lw=1)
ax.plot([0, x_range[-1]], [100]*2, ls='--', zorder=0, color='black', lw=1)
ax.set_xticks([])
# ax.set_xticklabels(['', '', '', ''])
ax.set_ylim([70, 160])
ax.set_yticks([70, 100, 130, 160])
ax.set_yticklabels([70, 100, 130, 160], fontsize=10)
rfv.despine(ax=ax, keep=['left'])
# ax2 = ax.twinx()
ax2.spines['left'].set_visible(True)
for trace in [trace[:min(num_frames)] for trace in lfp_traces]:
    ax2.plot(x_range, trace, c='black', lw=0.75, alpha=0.4)
ax2.set_ylim([-6.5, 1.3])
ax2.set_yticks([])
ax2.set_xticks([0, 10, 20, 30])
ax2.set_xticklabels([0, 10, 20, 30], fontsize=10)
rfv.despine(ax=ax2, remove=['top', 'right', 'left'])
rfv.add_scale_bar(ax=ax2, length=1, bartype='|', text='1mV', loc=(ax2.get_xlim()[1] + 1, -0.4), text_offset=[-0.4, 0], fs=10, lw=1.25)
ax.set_ylim([60, 160])
ax.set_ylabel('Mean FOV Ca$^{2+}$\n(% dFF)', fontsize=10)
# ax2.set_ylabel('LFP (mV)', fontsize=10)
ax2.set_xlabel('Time (secs)', fontsize=10)
# add lfp traces

rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0.01, x_adjust=0.13)
ax.set_ylim([70, 160])
ax.set_xlim([0, 30])
ax2.set_xlim([0, 30])
ax.margins(x=10, y=0)
ax2.margins(x=10, y=0)




# %% B - quantification of post-sz norm mean Flu of each seizure


mean_post_seizure_termination_all = Results.meanFOV_post_sz_term

plot_bar_with_points([mean_post_seizure_termination_all], x_tick_labels=[''], bar=False, ax=axes['B'][0], show=False, y_label='Mean FOV Ca$^{2+}$\n(% dFF)',
                     fig=fig, ylims=[60, 110], s=50, colors=['cornflowerblue'], alpha=1, fontsize=10)
rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0.01, x_adjust=0.12)


# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/suppl-sz-term-meanFOVsignal-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/suppl-sz-term-meanFOVsignal-RF.svg")

fig.show()


# %% MEASUREMENTS FUNCTIONS BELOW


# inspectExperimentMeanFOVandLFP(run_post=True)


# CHECKING POST-ICTAL SUPPRESSION PLOTS FOR ALL EXPERIMENTS
@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, allow_rerun=False, skip_trials=['PS07 t-011'])
def __photostim_timed_postictal(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    # inspectExperimentMeanFOVandLFP(exp_run=expobj.t_series_name)

    pre_termination_limit = -2  # seconds
    post_ictal_limit = 30  # seconds

    pre_stim_limit = 1  # seconds
    post_stim_limit = 2  # seconds

    lfp_trace = expobj.lfp_signal[expobj.frame_clock][:expobj.n_frames]

    lfp_traces = []
    num_frames = []
    post_ictal_frames = []
    for i, (onset, offset) in enumerate(zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets)):
        skip = False
        if expobj.t_series_name in [*ExpMetainfo.alloptical.post_ictal_exclude_sz]:
            if i in ExpMetainfo.alloptical.post_ictal_exclude_sz[expobj.t_series_name]:
                skip = True
        if not skip:
            frames = list(np.arange(int(offset - expobj.getFrames(pre_termination_limit)), int(offset + expobj.getFrames(post_ictal_limit))))
            post_ictal_frames.extend(frames)

    for i, stimframe in enumerate(expobj.stim_start_frames):
        if stimframe in post_ictal_frames:
            frames = np.arange(int(stimframe) - expobj.getFrames(pre_stim_limit), int(stimframe) + expobj.getFrames(post_stim_limit))
            lfp_traces.append(lfp_trace[frames])
            num_frames.append(len(frames))

    x_range = np.linspace(-pre_stim_limit, post_stim_limit, min(num_frames))
    mean_ = np.mean([trace[:min(num_frames)] for trace in lfp_traces], axis=0)
    sem_ = sem([trace[:min(num_frames)] for trace in lfp_traces])

    f, ax = plt.subplots(figsize=(5, 3))
    for trace in lfp_traces:
        trace_ = trace[:min(num_frames)]
        # subtract the mean from each element in the array
        trace_centered = [x - sum(trace_) / len(trace_) for x in trace_]
        trace_detrended = signal.detrend(trace_centered)

        # trace_centered = trace_
        # ax.plot(x_range, trace_centered, alpha=0.1, color='black', lw=1)
        ax.plot(x_range, trace_detrended, alpha=0.1, color='black', lw=1)
        # f.show()
    # ax.plot(x_range, mean_, c='black', lw=0.5)
    # ax.fill_between(x_range, mean_ - sem_, mean_ + sem_, alpha=0.3)
    ax.set_ylim([-4, 4])
    ax.set_ylabel('Mean LFP timed')
    ax.set_xlabel('Time (secs)')
    ax.set_title(f"{expobj.t_series_name}, photostim timed lfp traces\n0-centered, detrended", fontsize=8)
    f.show()

    return mean_, lfp_traces

# __photostim_timed_postictal()


# %% archive:

# CHECKING POST-ICTAL SUPPRESSION PLOTS FOR ALL EXPERIMENTS
@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=1, allow_rerun=True, skip_trials=['PS07 t-011'])
def __function(**kwargs):
    expobj: Post4ap = kwargs[
        'expobj']  # ; plt.plot(expobj.meanRawFluTrace, lw=0.5); plt.suptitle(expobj.t_series_name); plt.show()

    # inspectExperimentMeanFOVandLFP(exp_run=expobj.t_series_name)

    pre_termination_limit = 10  # seconds
    post_ictal_limit = 40  # seconds

    mean_interictal = np.median([expobj.meanRawFluTrace[x] for x in range(expobj.n_frames) if (x not in expobj.seizure_frames and x not in expobj.photostim_frames and x < expobj.photostim_frames[-1])])

    lfp_trace = expobj.lfp_signal[expobj.frame_clock][:expobj.n_frames]

    lfp_traces = []
    traces = []
    decay_constants = []
    num_frames = []
    for i, (onset, offset) in enumerate(zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets)):
        # onset2 = expobj.seizure_lfp_onsets[i+1]
        skip = False
        if expobj.t_series_name in [*ExpMetainfo.alloptical.post_ictal_exclude_sz]:
            if i in ExpMetainfo.alloptical.post_ictal_exclude_sz[expobj.t_series_name]:
                skip = True
        if not skip:
            frames = np.arange(int(offset - expobj.getFrames(pre_termination_limit)), int(offset + expobj.getFrames(post_ictal_limit)))
            frames = [frame for frame in frames if frame not in expobj.photostim_frames]  # filter out photostim frames - photostim artifact spiking imaging signal
            try:
                if onset > 0 and offset + int(expobj.getFrames(post_ictal_limit)) < expobj.n_frames:
                    pre_onset_frames = np.arange(int(onset - expobj.getFrames(20)), int(onset - expobj.getFrames(10)))
                    pre_onset_frames = [frame for frame in pre_onset_frames if frame not in expobj.photostim_frames]
                    pre_onset = expobj.meanRawFluTrace[pre_onset_frames]
                    pre_mean = np.mean(pre_onset)

                    # pre_onset_frames2 = np.arange(int(onset2 - expobj.getFrames(20)), int(onset2 - expobj.getFrames(10)))
                    # pre_onset_frames2 = [frame for frame in pre_onset_frames2 if frame not in expobj.photostim_frames]
                    # pre_onset2 = expobj.meanRawFluTrace[pre_onset_frames2]
                    # pre_mean2 = np.mean(pre_onset2)

                    trace = expobj.meanRawFluTrace[frames]
                    lfp_traces_ = lfp_trace[frames]

                    trace_norm = (trace / mean_interictal) * 100  # normalize trace to the selected mean of the interictal period

                    # trace_norm = (trace / pre_mean) * 100
                    # trace_norm = (trace / pre_mean2) * 100  # TRYING TO NORMALIZE THE PRE-ONSET BASELINE FOR THE UPCOMING SEIZURES!
                    # ax.plot(trace_norm, lw=0.3)

                    lfp_traces.append(lfp_traces_)
                    traces.append(trace_norm)
                    num_frames.append(len(frames))

                    # retrieve the decay constant
                    _trace_flipped = trace_norm * -1
                    _trace_corrected = convert_to_positive(_trace_flipped)
                    max_index = _trace_flipped.argmax()
                    # plt.plot(_trace_corrected); plt.show()
                    # plt.plot(_trace_corrected[max_index:]); plt.show()

                    # decay_c = calc_decay_constant(arr= _trace_flipped[max_index:], signal_rate = expobj.fps)
                    # decay_c2 = decay_constant(arr= trace_norm * -1, signal_rate = expobj.fps)

                    decay_c = decay_constant_logfit_method(arr = _trace_corrected[max_index:])
                    timesc = decay_timescale(arr=_trace_corrected[max_index:], signal_rate=expobj.fps, decay_constant=decay_c)

                    decay_constants.append(timesc)


            except:
                pass

    x_range = np.linspace(-pre_termination_limit, post_ictal_limit, min(num_frames))
    mean_ = np.mean([trace[:min(num_frames)] for trace in traces], axis=0)
    sem_ = sem([trace[:min(num_frames)] for trace in traces])

    _trace_flipped = mean_ * -1
    # _trace_corrected = convert_to_positive(_trace_flipped)
    _trace_corrected = (100 - mean_)
    # plt.plot(_trace_corrected); plt.show()
    max_index = _trace_corrected.argmax()
    below_zero_index = np.where(_trace_corrected[max_index:] < 0)[0][0] + max_index if np.any(_trace_corrected[max_index:] < 0) else -1  # only look at values that are positive

    # plt.plot(_trace_corrected); plt.show()
    # plt.plot(_trace_corrected[max_index:below_zero_index]); plt.show()

    decay_c_of_mean = decay_constant_logfit_method(arr=_trace_corrected[max_index:below_zero_index])
    timesc_of_mean = np.round(decay_timescale(arr=_trace_corrected[max_index:below_zero_index], signal_rate=expobj.fps, decay_constant=decay_c_of_mean), 2)

    if timesc_of_mean is np.nan:
        print('debugggg why the timescale isnot returning a value')

    f, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x_range, mean_, c='black', lw=0.5)
    ax.fill_between(x_range, mean_ - sem_, mean_ + sem_, alpha=0.3)
    ax.axhline(y=100)
    # ax.set_ylim([30, 400])
    ax.set_ylabel('Mean FOV Flu \n (norm. to interictal)')
    ax.set_xlabel('Time (secs)')
    ax.set_title(f"{expobj.t_series_name}, timescale of decay: {timesc_of_mean}secs", fontsize=8)
    # plot lfp traces
    ax2 = ax.twinx()
    ax2.set_ylabel('LFP (mV)')
    for trace in [trace[:min(num_frames)] for trace in lfp_traces]:
        ax2.plot(x_range, trace, c='black', lw=0.5, alpha=0.4)
    ax.spines['right'].set_visible(True)
    f.show()

    return mean_, lfp_traces, timesc_of_mean

# _return = __function()
# mean_post_seizure_termination_all, lfp_traces, timescales = _return[0], _return[1], _return[2]



# %% troubleshooting calculating decay constant
import numpy as np

def calc_decay_constant(signal: np.ndarray, signal_rate = 1):
    """Calculate decay constant of an input array.

    ChatGPT, 2022-12-04
    """
    # Calculate the natural log of the ratio of consecutive elements in the array
    log_ratios = np.log(arr[1:] / signal[:-1])

    # Calculate the difference between consecutive elements in the array
    time_intervals = np.diff(signal) / signal_rate  # convert index # to time units based on the signal collection rate

    # Divide the log ratios by the time intervals to get the decay constant
    decay_constants = np.divide(log_ratios, time_intervals)

    # Calculate the average decay constant
    avg_decay_constant = np.mean(decay_constants)

    return avg_decay_constant


def calculate_decay_constant(signal):
    # Calculate the mean of the signal array
    mean = np.mean(signal)

    # Calculate the difference between each value in the signal array and the mean
    diff = signal - mean

    # Calculate the sum of the squared differences
    ssd = np.sum(diff ** 2)

    # Calculate the decay constant
    decay_constant = 1 / ssd

    return decay_constant


def decay_constant_logfit_method(arr):
    """use the polyfit on the logarithm of the signal to calculate the decay coefficient

    >>> r = 0.5
    >>> a = 10
    >>> n = 10
    >>> arr = np.array([a*np.exp((-r)*i) for i in range(n)])
    >>> decay_constant = decay_constant_logfit_method(arr=arr)
    """
    coeffs = np.polyfit(range(n), np.log(arr), 1)
    decay_constant = -coeffs[0]
    return decay_constant


def decay_timescale(arr, decay_constant = None, signal_rate = 1):
    """
    Calculation of the decay timescale (optionally adjusting for signal collection data rate).

    :param arr:
    :param decay_constant:
    :param signal_rate:
    :return:

    >>> r = 0.5
    >>> a = 10
    >>> n = 10
    >>> arr = np.array([a*np.exp((-r)*i) for i in range(n)])
    >>> decay_constant = decay_constant_logfit_method(arr=arr)
    >>> decay_timescale(arr=arr, decay_constant=decay_constant, signal_rate=30)
    """

    max_value = np.max(arr)

    if decay_constant is None:
        decay_constant = decay_constant_logfit_method(arr=arr)

    timescale = -(1 / decay_constant) * np.log(1 - 1/np.e)
    half_life = -(1 / decay_constant) * np.log(0.5)

    plot = False
    if plot:
        time_steps = np.arange(0, len(arr))
        decay = max_value * np.exp(-decay_constant * time_steps)
        plt.plot(decay)
        plt.axhline(arr.max() * 0.5)
        plt.axvline(half_life)
        plt.suptitle('half life')
        plt.show()


        plt.plot(decay)
        plt.axhline(arr.max() * (1 - 1/np.e))
        plt.axvline(timescale)
        plt.suptitle('timescale value')
        plt.show()

    return timescale / signal_rate




def decay_constant(arr: np.ndarray, threshold: float = None, signal_rate: Union[float, int] = 1):
    """measure the timeconstant of decay of a signal array.
    If signal rate is provided, will return in units of time, otherwise will return as the index of the array.
    """
    if not type(arr) is np.ndarray:
        raise TypeError('provide `arr` input as type = np.array')
    max_value = arr.max()  # peak Flu value after stim
    max_index = arr.argmax()  # peak Flu value after stim
    threshold = np.exp(-1) * max_value if not threshold else threshold  # set threshold to be at 1/e x peak
    try:
        x_ = np.where(arr[max_index:] < threshold)[0][0]  # find index AFTER the index of the max value of the trace, where the trace decays to the threshold value
        return x_ / signal_rate  # convert frame # to time
    except Exception:
        print(f'Could not find decay below the maximum value of the trace provided. max: {max_value}, max index: {max_index}, decay threshold: {threshold}')






def calc_decay_constant(arr: np.ndarray, signal_rate: Union[float, int] = 1):
    """Calculate decay constant of an input array.

    ChatGPT, 2022-12-04
    """
    # Calculate the natural log of the ratio of consecutive elements in the array
    log_ratios = np.log(arr[1:] / arr[:-1])

    # Calculate the difference between consecutive elements in the array
    time_intervals = np.diff(arr) / signal_rate  # convert index # to time units based on the signal collection rate

    # Divide the log ratios by the time intervals to get the decay constant
    decay_constants = np.divide(log_ratios, time_intervals)

    # Calculate the average decay constant
    avg_decay_constant = np.mean(decay_constants)

    return avg_decay_constant


def decay_timescale(arr, decay_constant = None, signal_rate: Union[float, int] = 1):
    max_value = np.max(arr)

    if decay_constant is None:
        decay_constant = calc_decay_constant(arr=arr, signal_rate=signal_rate)
    threshold = 1 / np.e * max_value  # set threshold as: 1/e x max

    half_life = -(1 / decay_constant) * np.log(0.5)
    time_steps = np.arange(0, len(arr))
    # time_steps = np.linspace(0, len(arr)/signal_rate, len(arr))
    decay = max_value * np.exp(-decay_constant * time_steps)
    plt.plot(decay); plt.show()



    timescale = -(1 / decay_constant) * np.log(threshold)
    return timescale / signal_rate




