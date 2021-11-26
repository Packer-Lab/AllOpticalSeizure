# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS - FOCUS ON SLM TARGETS!
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal
import statsmodels.api
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj
import tifffile as tf
from skimage.transform import resize
from mpl_toolkits import mplot3d

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-24'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None

expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013')

# expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre e.1')  # PLACEHOLDER IMPORT OF EXPOBJ TO MAKE THE CODE WORK


#
"""######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


# %% sys.exit()
"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


# %% 0) plot representative experiment plot for stim responses - showing run_pre4ap_trials and run_post4ap_trials

# PRE4AP TRIAL
i = allopticalResults.pre_4ap_trials[0]
j = 0
pre4ap_prep = i[j][:-6]
pre4ap_trial = i[j][-5:]

# POST4AP TRIAL
i = allopticalResults.post_4ap_trials[0]
j = 0
post4ap_prep = i[j][:-6]
post4ap_trial = i[j][-5:]

trials_run = [f"{pre4ap_prep} {pre4ap_trial}", f"{post4ap_prep} {post4ap_trial}"]

@aoutils.run_for_loop_across_exps(trials_run=trials_run)
def plot_peristim_avg_traces_alltargets(to_plot='delta dF', **kwargs):

    expobj = kwargs['expobj'] if 'expobj' in kwargs.keys() else KeyError('need to provide expobj as keyword argument')
    if to_plot == 'dFstdF':
        arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dfstdF_avg])
        # arr = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])
        y_label = 'dFstdF (normalized to prestim period)'
        y_lims = [-0.25, 1.75]
    elif to_plot == 'delta dF':
        # arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg])
        arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dffAvg])
        y_label = 'delta dF (relative to prestim)'
        y_lims = [-15, 75]
    else:
        raise TypeError('unrecognized call to to_plot! (valid options are `dFstdF` or `delta dF`) ')

    aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=expobj.pre_stim / expobj.fps, post_stim_sec=expobj.post_stim / expobj.fps,
                                  title=(f'{expobj.metainfo["animal prep."]} {expobj.metainfo["trial"]} - trace dFF - photostim targets'), figsize=[8, 6], y_label=y_label,
                                  x_label='Time', y_lims=y_lims)

    x_range = [np.linspace(0, arr.shape[1] / expobj.fps, arr.shape[1])] * arr.shape[0]
    pj.make_general_plot(data_arr=arr, x_range=x_range, v_span=(expobj.pre_stim / expobj.fps, (expobj.pre_stim + expobj.stim_duration_frames)/expobj.fps),
                         y_label=y_label, x_label='Time (secs)', title=(f'{expobj.metainfo["animal prep."]} {expobj.metainfo["trial"]} - trace dFF - photostim targets'),
                         fontsize=12, figsize=(4,4))


plot_peristim_avg_traces_alltargets(to_plot='delta dF')



# %% 1) plot peri-photostim avg traces for all trials analyzed to make sure they look alright -- plot as little postage stamps


# PRE4AP TRIAL
i = allopticalResults.pre_4ap_trials[0]
j = 0
pre4ap_prep = i[j][:-6]
pre4ap_trial = i[j][-5:]

# POST4AP TRIAL
i = allopticalResults.post_4ap_trials[0]
j = 0
post4ap_prep = i[j][:-6]
post4ap_trial = i[j][-5:]

trials_run = [f"{pre4ap_prep} {pre4ap_trial}", f"{post4ap_prep} {post4ap_trial}"][0]

@aoutils.run_for_loop_across_exps(trials_run=[trials_run])
def plot_postage_stamps_photostim_traces(to_plot='delta dF',**kwargs):
    expobj = kwargs['expobj'] if 'expobj' in kwargs.keys() else KeyError('need to provide expobj as keyword argument')

    if to_plot == 'delta dF':
        responses = expobj.responses_SLMtargets_tracedFF
        hits = expobj.hits_SLMtargets_tracedFF
        trace_snippets = expobj.SLMTargets_tracedFF_stims_dff  # TODO confirm that the pre-stim period has mean of 0 for all these traces!
        stimsuccessrates = expobj.StimSuccessRate_SLMtargets_tracedFF
        y_label = 'delta dF'
    elif to_plot == 'dFF':
        responses = expobj.responses_SLMtargets
        hits = expobj.hits_SLMtargets
        trace_snippets = expobj.SLMTargets_stims_dff
        stimsuccessrates = expobj.StimSuccessRate_SLMtargets
        y_label = '% dFF'
    else:
        raise ValueError('must provide to_plot as either `dFF` or `delta dF`')

    responses_magnitudes_successes = {}
    response_traces_successes = {}
    responses_magnitudes_failures = {}
    response_traces_failures = {}

    nrows = expobj.n_targets_total // 4
    if expobj.n_targets_total % 4 > 0:
        nrows += 1
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3),
                            constrained_layout=True)
    counter = 0
    axs[0, 0].set_xlabel('Time (secs)')
    axs[0, 0].set_ylabel(y_label)

    for cell in range(trace_snippets.shape[0]):
        a = counter // 4
        b = counter % 4
        alpha = 1 * (stimsuccessrates[cell] / 100)
        print(f'plotting target #{counter}, success rate: {stimsuccessrates[cell]}\n')
        if cell not in responses_magnitudes_successes.keys():
            responses_magnitudes_successes[cell] = []
            response_traces_successes[cell] = np.zeros((trace_snippets.shape[-1]))
            responses_magnitudes_failures[cell] = []
            response_traces_failures[cell] = np.zeros((trace_snippets.shape[-1]))

        success_stims = np.where(hits.loc[cell] == 1)
        fail_stims = np.where(hits.loc[cell] == 0)

        x_range = np.linspace(0, len(trace_snippets[cell][0]) / expobj.fps, len(trace_snippets[cell][0]))

        # success_stims = np.where(expobj.responses_SLMtargets.loc[cell] >= 0.1 * 100)
        # fail_stims = np.where(expobj.responses_SLMtargets.loc[cell] < 0.1 * 100)
        for i in success_stims[0]:
            trace = trace_snippets[cell][i]
            axs[a, b].plot(x_range, trace, color='skyblue', zorder=2, alpha=0.05)

        for i in fail_stims[0]:
            trace = trace_snippets[cell][i]
            axs[a, b].plot(x_range, trace, color='gray', zorder=3, alpha=0.05)

        if len(success_stims[0]) > 5:
            success_avg = np.nanmean(trace_snippets[cell][success_stims], axis=0)
            axs[a, b].plot(x_range, success_avg, color='navy', linewidth=2, zorder=4, alpha=1)
        if len(fail_stims[0]) > 5:
            failures_avg = np.nanmean(trace_snippets[cell][fail_stims], axis=0)
            axs[a, b].plot(x_range, failures_avg, color='black', linewidth=2, zorder=4, alpha=1)
        axs[a, b].set_ylim([-0.2 * 100, 1.0 * 100])
        axs[a, b].text(0.98, 0.97, f"Success rate: {stimsuccessrates[cell]:.0f}%",
                       verticalalignment='top', horizontalalignment='right',
                       transform=axs[a, b].transAxes, fontweight='bold',
                       color='black')
        axs[a, b].margins(0)
        axs[a, b].axvspan(expobj.pre_stim / expobj.fps, (expobj.pre_stim + expobj.stim_duration_frames) / expobj.fps, color='mistyrose',
                          zorder=0)

        counter += 1
    fig.suptitle(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - {len(trace_snippets)} targets",
                 y = 0.995)
    # fig.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s_individual targets dFF.png' % (date, j[:-6], date, j))
    fig.tight_layout(pad=1.8)
    fig.show()

plot_postage_stamps_photostim_traces()


# %% 2) BAR PLOT FOR PHOTOSTIM RESPONSE MAGNITUDE B/W PRE AND POST 4AP TRIALS - TODO plot delta (trace dFF) responses

y_label = 'delta(trace_dFF)'
to_process = f"mean response ({y_label} all targets)"

pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses['prep_trial'] == trial, to_process].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

post4ap_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses['prep_trial'] == trial, to_process].values[0] for trial in i]
    post4ap_response_magnitude.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_response_magnitude, post4ap_response_magnitude], paired=True, shrink_text=0.9,
                        colors=['gray', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.5, #ylims=[-50, 100],
                        x_tick_labels=['pre-4ap', 'post-4ap'], title=f"Mean {y_label}", y_label=y_label, title_pad=10)


# %% 3) BAR PLOT FOR PHOTOSTIM RESPONSE RELIABILITY B/W PRE AND POST 4AP TRIALS

plot = '(>0.3 dF/stdF)'
# plot = '(>10 delta(trace_dFF))'
to_process = f'mean reliability {plot}'


pre4ap_reliability = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, to_process].values[0] for trial in i]
    pre4ap_reliability.append(np.mean(x))

post4ap_reliability = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, to_process].values[0] for trial in i]
    post4ap_reliability.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_reliability, post4ap_reliability], paired=True,
                        colors=['gray', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'post-4ap'], title=f'Avg. Reliability {plot}',
                        y_label='% success rate of photostim')






# %% 4) plot responses of SLM TARGETS in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
# - with option to plot only successful or only failure stims!

# ## 4.1) PRE-4AP TRIALS

avgtraces = {'pre4ap': {}}
@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True) #, trials_run=pj.flattenOnce(allopticalResults.pre_4ap_trials[:3]))
def plot_avg_stim_traces_pre4apexps(process='dfstdf', to_plot='successes', avg_only=True, f=None, ax=None,
                                    **kwargs):
    """
    Plot the average photostim response for all experiments specified.

    :param process: use for plotting either `delta(trace_dFF)` processed responses or `dfstdf` responses
    :param to_plot: use for plotting either `successes` stim responses or `failures` stim responses
    :param avg_only: plot avg only for each expobj
    :param f, ax: be sure to make and provide a matplotlib subplots figure and a axis objects to collect the plot
    :param kwargs:
        'expobj': must be imported prior to running function and provided in the function call
    :return: None

    Examples:
    >>> f, ax = plt.subplots(figsize=[5, 4])
    >>> plot_avg_stim_traces_pre4apexps()
    >>> f.show()
    """

    # if not hasattr(expobj, 'traces_SLMtargets_successes_avg'):
    #     print('running .calculate_SLMTarget_SuccessStims method for expobj of %s, %s [1.1.1]' % (prep, trial))
    #     expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
    #     expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
    #     expobj.traces_SLMtargets_failures_avg = \
    #         expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)

    expobj = kwargs['expobj']

    print(f"|- Plotting avg traces of {to_plot} from expobj")

    if to_plot == 'successes':
        if process == 'delta(trace_dFF)':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_tracedFF_successes_avg[key] for key in
                            expobj.traces_SLMtargets_tracedFF_successes_avg.keys()])
            y_lims = [-30, 100]
        elif process == 'dfstdf':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_successes_avg_dfstdf[key] for key in
                            expobj.traces_SLMtargets_successes_avg_dfstdf.keys()])
            y_lims = [-0.3, 3.0]
    elif to_plot == 'failures':
        if process == 'delta(trace_dFF)':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_tracedFF_failures_avg[key] for key in
                            expobj.traces_SLMtargets_tracedFF_failures_avg.keys()])
            y_lims = [-30, 100]
        elif process == 'dfstdf':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_failures_avg_dfstdf[key] for key in
                            expobj.traces_SLMtargets_failures_avg_dfstdf.keys()])
            y_lims = [-0.3, 3.0]


    # prepare data for plotting
    y_label = '% dFF (normalized to prestim period)'
    x_label = 'Time (secs)'
    pre_stim_sec = expobj.pre_stim / expobj.fps
    post_stim_sec = expobj.post_stim / expobj.fps

    if len(array_to_plot) > 0:
        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 3 frames as stimduration
                trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + expobj.post_stim])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur_sec = 3 / expobj.fps
            title = f"{process} {to_plot} stims only, all exps. - avg. of targets - run_pre4ap_trials"
        else:
            data_traces = array_to_plot
            stim_dur_sec = expobj.stim_duration_frames / expobj.fps
            title = f"{process} {to_plot} stims only - avg. responses of photostim targets - run_pre4ap_trials stims - {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"


        f, ax, avg = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj, stim_duration=stim_dur_sec, y_lims=y_lims,
                                                   pre_stim_sec=pre_stim_sec, post_stim_sec=post_stim_sec, avg_only=avg_only, title=title,
                                                   y_label=y_label, x_label=x_label, fig=f, ax=ax, show=False, alpha=1, pad=30)


        kwargs['dffAvgTraces'].append(avg) if 'dffAvgTraces' in kwargs.keys() else None
        print('|- shape of dFF array: ', data_traces.shape, ' [1.1.3]')
        print(f'|- len of dffAvgTraces collection: {len(kwargs["dffAvgTraces"])}') if 'dffAvgTraces' in kwargs.keys() else None
        # print(f'|- length of dffTraces list {len(dffTraces)}')

        # return kwargs['dffAvgTraces'] if 'dffAvgTraces' in kwargs.keys() else None
        return avg if 'dffAvgTraces' in kwargs.keys() else None

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['pre4ap']['successes_delta(trace_dFF)'] = plot_avg_stim_traces_pre4apexps(to_plot='successes', process='delta(trace_dFF)', dffAvgTraces=[],
                                                                                    f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['pre4ap']['failures_delta(trace_dFF)'] = plot_avg_stim_traces_pre4apexps(to_plot='failures', process='delta(trace_dFF)', dffAvgTraces=[],
                                                                                   f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['pre4ap']['successes_dfstdf'] = plot_avg_stim_traces_pre4apexps(to_plot='successes', process='dfstdf', dffAvgTraces=[],
                                                                          f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['pre4ap']['failures_dfstdf'] = plot_avg_stim_traces_pre4apexps(to_plot='failures', process='dfstdf', dffAvgTraces=[],
                                                                         f=f, ax=ax)
f.show()


allopticalResults.avgTraces['pre4ap'] = avgtraces['pre4ap']
allopticalResults.save()



# %% 4.2) PLOT AVG PERISTIM RESPONSES OF SLM TARGETS ACROSS POST-4AP TRIALS - IN SZ STIMS - EXCLUDE STIMS/CELLS INSIDE SZ BOUNDARY


avgtraces = {'insz': {}}
@aoutils.run_for_loop_across_exps(run_post4ap_trials=True) #, trials_run=pj.flattenOnce(allopticalResults.pre_4ap_trials[:3]))
def plot_avg_stim_traces_inszexps(process='dfstdf', to_plot='successes', avg_only=True, f=None, ax=None,
                                  **kwargs):
    """
    Plot the average photostim response for all experiments specified.

    :param process: use for plotting either `delta(trace_dFF)` processed responses or `dfstdf` responses
    :param to_plot: use for plotting either `successes` stim responses or `failures` stim responses
    :param avg_only: plot avg only for each expobj
    :param f, ax: be sure to make and provide a matplotlib subplots figure and a axis objects to collect the plot
    :param kwargs:
        'expobj': must be imported prior to running function and provided in the function call
    :return: None

    Examples:
    >>> f, ax = plt.subplots(figsize=[5, 4])
    >>> plot_avg_stim_traces_pre4apexps()
    >>> f.show()
    """

    # if not hasattr(expobj, 'traces_SLMtargets_successes_avg'):
    #     print('running .calculate_SLMTarget_SuccessStims method for expobj of %s, %s [1.1.1]' % (prep, trial))
    #     expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
    #     expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
    #     expobj.traces_SLMtargets_failures_avg = \
    #         expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)

    expobj = kwargs['expobj']

    print(f"|- Plotting avg traces of {to_plot} from expobj")

    if to_plot == 'successes':
        if process == 'delta(trace_dFF)':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_tracedFF_successes_avg[key] for key in
                            expobj.insz_traces_SLMtargets_tracedFF_successes_avg.keys()])
            y_lims = [-30, 100]
        elif process == 'dfstdf':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_successes_avg_dfstdf[key] for key in
                            expobj.insz_traces_SLMtargets_successes_avg_dfstdf.keys()])
            y_lims = [-0.3, 3.0]
    elif to_plot == 'failures':
        if process == 'delta(trace_dFF)':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_tracedFF_failures_avg[key] for key in
                            expobj.insz_traces_SLMtargets_tracedFF_failures_avg.keys()])
            y_lims = [-30, 100]
        elif process == 'dfstdf':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_failures_avg_dfstdf[key] for key in
                            expobj.insz_traces_SLMtargets_failures_avg_dfstdf.keys()])
            y_lims = [-0.3, 3.0]


    # prepare data for plotting
    y_label = '% dFF (normalized to prestim period)'
    x_label = 'Time (secs)'
    pre_stim_sec = expobj.pre_stim / expobj.fps
    post_stim_sec = expobj.post_stim / expobj.fps

    if len(array_to_plot) > 0:
        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 3 frames as stimduration
                trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + expobj.post_stim])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur_sec = 3 / expobj.fps
            title = f"{process} {to_plot} stims only, all exps. - avg. of targets - post4ap insz stims"
        else:
            data_traces = array_to_plot
            stim_dur_sec = expobj.stim_duration_frames / expobj.fps
            title = f"{process} {to_plot} stims only - avg. responses of photostim targets - post4ap insz stims - {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"


        f, ax, avg = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj, stim_duration=stim_dur_sec, y_lims=y_lims,
                                                   pre_stim_sec=pre_stim_sec, post_stim_sec=post_stim_sec, avg_only=avg_only, title=title,
                                                   y_label=y_label, x_label=x_label, fig=f, ax=ax, show=False, alpha=1, pad=30)


        kwargs['dffAvgTraces'].append(avg) if 'dffAvgTraces' in kwargs.keys() else None
        print('|- shape of dFF array: ', data_traces.shape, ' [1.1.3]')
        print(f'|- len of dffAvgTraces collection: {len(kwargs["dffAvgTraces"])}') if 'dffAvgTraces' in kwargs.keys() else None
        # print(f'|- length of dffTraces list {len(dffTraces)}')

        # return kwargs['dffAvgTraces'] if 'dffAvgTraces' in kwargs.keys() else None
        return avg if 'dffAvgTraces' in kwargs.keys() else None


f, ax = plt.subplots(figsize=[5, 4])
avgtraces['insz']['successes_delta(trace_dFF)'] = plot_avg_stim_traces_inszexps(to_plot='successes', process='delta(trace_dFF)', dffAvgTraces=[],
                                                                                f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['insz']['failures_delta(trace_dFF)'] = plot_avg_stim_traces_inszexps(to_plot='failures', process='delta(trace_dFF)', dffAvgTraces=[],
                                                                               f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['insz']['successes_dfstdf'] = plot_avg_stim_traces_inszexps(to_plot='successes', process='dfstdf', dffAvgTraces=[],
                                                                      f=f, ax=ax)
f.show()

f, ax = plt.subplots(figsize=[5, 4])
avgtraces['insz']['failures_dfstdf'] = plot_avg_stim_traces_inszexps(to_plot='failures', process='dfstdf', dffAvgTraces=[],
                                                                     f=f, ax=ax)
f.show()


allopticalResults.avgTraces['insz'] = avgtraces['insz']
allopticalResults.save()


# %% ## 4.3) POST-4AP TRIALS (OUT SZ STIMS)

run_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True
to_plot = 'successes'  # use for plotting either 'successes' stim responses or 'failures' stim responses
re_plot = True


dffTraces_outsz = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.post_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[1]
        # j = 0
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.3.1]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if 'post' in expobj.metainfo['exptype']:
            if run_processing:
                aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                            plots=False, force_redo=False)
                expobj.save()

        if not hasattr(expobj, 'outsz_traces_SLMtargets_successes_avg') or run_processing:
            stims_outsz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            if len(stims_outsz_idx) > 0:
                print('|- calculating stim success rates (outsz) - %s stims [1.3.3]' % len(stims_outsz_idx))
                expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_traces_SLMtargets_successes_avg, \
                expobj.outsz_traces_SLMtargets_failures_avg = \
                    expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets,
                                                            stims_idx_l=stims_outsz_idx)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.outsz_traces_SLMtargets_successes_avg[key] for key in expobj.outsz_traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.outsz_traces_SLMtargets_failures_avg[key] for key in expobj.outsz_traces_SLMtargets_failures_avg.keys()])

        y_label = 'pct. dFF (normalized to prestim period)'

        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in np.asarray([expobj.outsz_traces_SLMtargets_successes_avg[key] for key in expobj.outsz_traces_SLMtargets_successes_avg.keys()]):
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 5 frames as stimduration
                trace_ = np.append(trace_, trace[-expobj.post_stim:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - out sz stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - out sz stims %s %s' % (to_plot, prep, trial)


        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj,
                                                 stim_duration=stim_dur, y_lims=[0, 50],
                                                 pre_stim_sec=0.25, exp_prestim=expobj.pre_stim, post_stim_sec=2.75, avg_only=avg_only,
                                                 title=title, y_label=y_label, x_label='Time (secs)', fig=f, ax=ax, show=False)

        print('|- shape of dFF array: ', data_traces.shape, ' [1.3.4]')
        dffTraces_outsz.append(d)

f.show()
allopticalResults.dffTraces_outsz = np.asarray(dffTraces_outsz)
allopticalResults.save()






# 4.4)
from scipy.interpolate import interp1d

traces = []
x_long = allopticalResults.dffTraces_outsz[0][1]
f, ax = plt.subplots(figsize=(6, 5))
for trace in allopticalResults.dffTraces_outsz:
    if len(trace[1]) < len(x_long):
        f2 = interp1d(trace[1], trace[2])
        trace_plot = f2(x_long)
        ax.plot(x_long, trace_plot, color='gray')
    else:
        trace_plot = trace[2]
        ax.plot(trace[1], trace_plot, color='gray')
    traces.append(trace_plot)
ax.axvspan(0.4, 0.48 + 3 / 30, alpha=1, color='tomato', zorder=3)  # where 30 == fps for the fastest imaging experiments
avgTrace = np.mean(np.array(traces), axis=0)
ax.plot(x_long, avgTrace, color='black', lw=3)
ax.set_title('avg of all targets per exp. for stims out_sz - each trace = t-series from allopticalResults.post_4ap_trials - dFF photostim',
             horizontalalignment='center', verticalalignment='top', pad=35, fontsize=13, wrap=True)
ax.set_xlabel('Time (secs)')
ax.set_ylabel('dFF (norm. to pre-stim F)')
f.show()


# %% 5.1) bar PLOT - COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ
## make bar plot using the collected response magnitudes
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response (hits, all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

outsz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response outsz (hits, all targets)'].values[0] for trial in i]
    outsz_response_magnitude.append(np.mean(x))

insz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response insz (hits, all targets)'].values[0] for trial in i]
    insz_response_magnitude.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_response_magnitude, outsz_response_magnitude, insz_response_magnitude], paired=True,
                        colors=['black', 'purple', 'red'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'outsz', 'insz'], title='Avg. Response magnitude of hits',
                        y_label='response magnitude (dFF)')

# %% 5.2) bar PLOT - COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ
## make bar plot using the collected response magnitudes
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response (failures, all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

outsz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response outsz (failures, all targets)'].values[0] for trial in i]
    outsz_response_magnitude.append(np.mean(x))

insz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response insz (failures, all targets)'].values[0] for trial in i]
    insz_response_magnitude.append(np.mean(x))


pj.plot_bar_with_points(data=[pre4ap_response_magnitude, outsz_response_magnitude, insz_response_magnitude], paired=False,
                        colors=['black', 'purple', 'red'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'outsz', 'insz'], title='Avg. Response magnitude of failures',
                        y_label='response magnitude (dFF)')


# %% 6.2.1-cd) PLOT - absolute stim responses vs. TIME to seizure onset - using trace dFF processed data - using new general scatter func code

"""todo for this analysis:
- average over targets for plot containing all exps
"""

# plotting of post_4ap zscore_stim_relative_to_sz onset
print(f"plotting averages from trials: {list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())}")

preps = np.unique([prep[:-6] for prep in allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys()])

exps = list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())

x_points = []
y_points = []
ax_titles = []
for i in range(len(preps)):
    print(i)
    for key in allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys():
        if preps[i] in key:
            print(key)
            sz_time = allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[key][0]
            responses = allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[key][1]
            x_points.append(sz_time)
            y_points.append(responses)
            ax_titles.append(key)

pj.make_general_scatter(x_list=pj.flattenOnce(x_points), y_data=pj.flattenOnce(y_points), ax_titles=ax_titles)


# %% 6.2-cd) PLOT - absolute stim responses vs. TIME to seizure onset - using trace dFF processed data

"""todo for this analysis:
- average over targets for plot containing all exps
"""

# plotting of post_4ap zscore_stim_relative_to_sz onset
print(f"plotting averages from trials: {list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())}")

preps = np.unique([prep[:-6] for prep in allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys()])

exps = list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())

xlabel = 'Time to closest seizure onset (secs)'
ylabel = 'avg dFF response'

## prep for large figure with individual experiments
ncols = 4
nrows = len(exps) // ncols
if len(exps) % ncols > 0:
    nrows += 1

fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[(ncols * 4), (nrows * 3)])
counter = 0
axs[0, 0].set_xlabel(xlabel)
axs[0, 0].set_ylabel(ylabel)

# prep for single small plot with all experiments
fig, ax = plt.subplots(figsize=(4, 3))
colors = pj.make_random_color_array(n_colors=len(preps))

for i in range(len(preps)):
    print(i)
    for key in allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys():
        if preps[i] in key:
            print(key)
            sz_time = allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[key][0]
            responses = allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[key][1]

            # xes = list(np.where(np.isnan(responses[0]))[0])
            # responses_to_plot = [responses[0][i] for i in range(len(responses)) if i not in xes]
            # sz_time_plot = [sz_time[0][i] for i in range(len(responses)) if i not in xes]
            # if len(responses_to_plot) > 1:
            #     print(f"plotting responses for {key}")

            # ax.scatter(x=sz_time_plot, y=responses_to_plot, facecolors=colors[i], alpha=0.2, lw=0)
            ax.scatter(x=sz_time, y=responses, facecolors=colors[i], alpha=0.2, lw=0)

            a = counter // ncols
            b = counter % ncols

            # make plot for individual key/experiment trial
            ax2 = axs[a, b]
            ax2.scatter(x=sz_time, y=responses, facecolors=colors[i], alpha=0.8, lw=0)
            ax2.set_xlim(-50, 50)
            ax2.set_title(f"{key}")
            counter += 1

ax.set_xlim(-50, 50)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

fig.suptitle(f"All exps, all targets relative to closest sz onset")
fig.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/responsescore-dFF-vs-szonset_time_allexps.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()

fig2.suptitle(f"all exps. individual")
fig2.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/responsescore-dFF-vs-szonset_time_individualexps.png"
print(f'\nsaving figure2 to {save_path_full}')
fig2.savefig(save_path_full)
fig2.show()


allopticalResults.save()

# %% 6.1.2) PLOT - zscore of stim responses vs. TIME to seizure onset

"""todo for this analysis:
- average over targets for plot containing all exps
"""

# plotting of post_4ap zscore_stim_relative_to_sz onset
print(f"plotting averages from trials: {list(allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim.keys())}")

preps = np.unique([prep[:-6] for prep in allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim.keys()])

exps = list(allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim.keys())

## prep for large figure with individual experiments
ncols = 4
nrows = len(exps) // ncols
if len(exps) % ncols > 0:
    nrows += 1

fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[(ncols * 4), (nrows * 3)])
counter = 0
axs[0, 0].set_xlabel('Time to closest seizure onset (secs)')
axs[0, 0].set_ylabel('responses (z scored)')

# prep for single small plot with all experiments
fig, ax = plt.subplots(figsize=(4, 3))
colors = pj.make_random_color_array(n_colors=len(preps))
for i in range(len(preps)):
    print(i)
    for key in allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim.keys():
        if preps[i] in key:
            print(key)
            sz_time = allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim[key][0]
            z_scores = allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim[key][1]
            ax.scatter(x=sz_time, y=z_scores, facecolors=colors[i], alpha=0.2, lw=0)

            a = counter // ncols
            b = counter % ncols

            # make plot for individual key/experiment trial
            ax2 = axs[a, b]
            ax2.scatter(x=sz_time, y=z_scores, facecolors=colors[i], alpha=0.8, lw=0)
            ax2.set_xlim(-300, 250)
            ax2.set_title(f"{key}")
            counter += 1

ax.set_xlim(-300, 250)
ax.set_xlabel('Time to closest seizure onset (secs)')
ax.set_ylabel('responses (z scored)')

fig.suptitle(f"All exps, all targets relative to closest sz onset")
fig.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/zscore-vs-szonset_time_allexps.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()

fig2.suptitle(f"all exps. individual")
fig2.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/zscore-vs-szonset_time_individualexps.png"
print(f'\nsaving figure2 to {save_path_full}')
fig2.savefig(save_path_full)
fig2.show()

# %% 6.1.1) PLOT - absolute stim responses vs. TIME to seizure onset

"""todo for this analysis:
- average over targets for plot containing all exps
"""

# plotting of post_4ap zscore_stim_relative_to_sz onset
print(f"plotting averages from trials: {list(allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim.keys())}")

preps = np.unique([prep[:-6] for prep in allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim.keys()])

exps = list(allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim.keys())

## prep for large figure with individual experiments
ncols = 4
nrows = len(exps) // ncols
if len(exps) % ncols > 0:
    nrows += 1

fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[(ncols * 4), (nrows * 3)])
counter = 0
axs[0, 0].set_xlabel('Time to closest seizure onset (secs)')
axs[0, 0].set_ylabel('response magnitude')

# prep for single small plot with all experiments
fig, ax = plt.subplots(figsize=(4, 3))
colors = pj.make_random_color_array(n_colors=len(preps))
for i in range(len(preps)):
    print(i)
    for key in allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim.keys():
        if preps[i] in key:
            print(key)
            sz_time = allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim[key][0]
            responses = allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim[key][1]
            # xes = list(np.where(np.isnan(responses[0]))[0])
            # responses_to_plot = [responses[0][i] for i in range(len(responses)) if i not in xes]
            # sz_time_plot = [sz_time[0][i] for i in range(len(responses)) if i not in xes]
            # if len(responses_to_plot) > 1:
            #     print(f"plotting responses for {key}")

            # ax.scatter(x=sz_time_plot, y=responses_to_plot, facecolors=colors[i], alpha=0.2, lw=0)
            ax.scatter(x=sz_time, y=responses, facecolors=colors[i], alpha=0.2, lw=0)

            a = counter // ncols
            b = counter % ncols

            # make plot for individual key/experiment trial
            ax2 = axs[a, b]
            ax2.scatter(x=sz_time, y=responses, facecolors=colors[i], alpha=0.8, lw=0)
            ax2.set_xlim(-300, 250)
            ax2.set_title(f"{key}")
            counter += 1

ax.set_xlim(-300, 250)
ax.set_xlabel('Time to closest seizure onset (secs)')
ax.set_ylabel('responses')

fig.suptitle(f"All exps, all targets relative to closest sz onset")
fig.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/responsescore-vs-szonset_time_allexps.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()

fig2.suptitle(f"all exps. individual")
fig2.tight_layout(pad=1.8)
save_path_full = f"{save_path_prefix}/responsescore-vs-szonset_time_individualexps.png"
print(f'\nsaving figure2 to {save_path_full}')
fig2.savefig(save_path_full)
fig2.show()


allopticalResults.save()




# %% archive-9.0-cd) zscore of stim responses vs. TIME to seizure onset - original code for single experiments
prep = 'RL108'
date = '2020-12-18'
trial = 't-013'
expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)
post_4ap_df = expobj.responses_SLMtargets_zscore

# transform the rows of the stims responses dataframe to relative time to seizure

stims = list(post_4ap_df.index)
stims_relative_sz = []
for stim_idx in stims:
    stim_frame = expobj.stim_start_frames[stim_idx]
    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
    stims_relative_sz.append(round(time_diff, 3))

cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x=post_4ap_df_zscore_stim_relative_to_sz.index, y=post_4ap_df_zscore_stim_relative_to_sz['avg'])
fig.show()