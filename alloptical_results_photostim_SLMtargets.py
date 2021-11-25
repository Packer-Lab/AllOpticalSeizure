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


# sys.exit()
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
prep = i[j][:-6]
trial = i[j][-5:]
print('\nprogress @ ', prep, trial, ' [1.1.0]')
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

to_plot = 'dFF'

if to_plot == 'dFstdF':
    arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dfstdF_avg])
    # arr = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])
    y_label = 'dFstdF (normalized to prestim period)'
    y_lims = [-0.25, 1.75]
elif to_plot == 'dFF':
    # arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg])
    arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dffAvg])
    y_label = 'dFF (normalized to prestim period)'
    y_lims = [-15, 75]
aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=expobj.pre_stim / expobj.fps, post_stim_sec=expobj.post_stim / expobj.fps,
                              title=(f'{prep} {trial} - trace dFF - photostim targets'), figsize=[4, 4], y_label=y_label,
                              x_label='Time', y_lims=y_lims)




#  POST4AP TRIAL
i = allopticalResults.post_4ap_trials[0]
j = 0
prep = i[j][:-6]
trial = i[j][-5:]
print('\nprogress @ ', prep, trial, ' [1.1.0]')
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

to_plot = 'dFF'

if to_plot == 'dFstdF':
    arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dfstdF_avg])
    # arr = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])
    y_label = 'dFstdF (normalized to prestim period)'
    y_lims = [-0.25, 1.75]
elif to_plot == 'dFF':
    # arr = np.asarray([i for i in expobj.SLMTargets_stims_dffAvg])
    arr = np.asarray([i for i in expobj.SLMTargets_tracedFF_stims_dffAvg])
    y_label = 'dFF (normalized to prestim period)'
    y_lims = [-15, 75]
aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=expobj.pre_stim / expobj.fps, post_stim_sec=expobj.post_stim / expobj.fps,
                              title=(f'{prep} {trial} - trace dFF - photostim targets'), figsize=[4, 4], y_label=y_label,
                              x_label='Time', y_lims=y_lims)


# %% 1) plot peri-photostim avg traces for all trials analyzed to make sure they look alright -- plot as little postage stamps

"""# plot avg of successes in green
# plot avg of failures in gray
# plot line at dF_stdF = 0.3
# add text in plot for avg dF_stdF value of successes, and % of successes

"""

# make one figure for each prep/trial (one little plot for each cell in that prep)
for exp in allopticalResults.pre_4ap_trials:
    # exp = ['RL108 t-009']
    calc_dff_stims = False
    for j in exp:
        if 'PS18' in j:
            date = allopticalResults.slmtargets_stim_responses.loc[
                allopticalResults.slmtargets_stim_responses['prep_trial'] == j, 'date'].values[0]
            pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
                date, j[:-6], date, j[-5:], date, j[-5:])  # specify path in Analysis folder to save pkl object

            expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)
            if calc_dff_stims:
                pass
                # print('\n Calculating stim success rates and response magnitudes [4.1] ***********')
                # expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                #     aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=10,
                #                                               stims_to_use=expobj.stim_start_frames)
                # expobj.save()

            # raw_traces_stims = expobj.SLMTargets_stims_raw

            # expobj.post_stim_response_window_msec = 500
            # expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)

            nrows = expobj.n_targets_total // 4
            if expobj.n_targets_total % 4 > 0:
                nrows += 1
            ncols = 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3),
                                    constrained_layout=True)
            counter = 0
            axs[0, 0].set_xlabel('Frames')
            axs[0, 0].set_ylabel('% dFF')

            responses_magnitudes_successes = {}
            response_traces_successes = {}
            responses_magnitudes_failures = {}
            response_traces_failures = {}

            for cell in range(expobj.SLMTargets_stims_dff.shape[0]):
                a = counter // 4
                b = counter % 4
                print('\n%s' % counter)
                if cell not in responses_magnitudes_successes.keys():
                    responses_magnitudes_successes[cell] = []
                    response_traces_successes[cell] = np.zeros((expobj.SLMTargets_stims_dff.shape[-1]))
                    responses_magnitudes_failures[cell] = []
                    response_traces_failures[cell] = np.zeros((expobj.SLMTargets_stims_dff.shape[-1]))

                # reliability = expobj.StimSuccessRate_SLMtargets[cell]
                # f, axs = plt.subplots(figsize=(5, 12), nrows=3)
                # fig, ax = plt.subplots(figsize=(3, 3))

                success_stims = np.where(expobj.responses_SLMtargets.loc[cell] >= 0.1 * 100)
                fail_stims = np.where(expobj.responses_SLMtargets.loc[cell] < 0.1 * 100)
                for i in success_stims[0]:
                    trace = expobj.SLMTargets_stims_dff[cell][i]
                    axs[a, b].plot(trace, color='skyblue', zorder=2, alpha=0.05)

                for i in fail_stims[0]:
                    trace = expobj.SLMTargets_stims_dff[cell][i]
                    axs[a, b].plot(trace, color='gray', zorder=3, alpha=0.05)

                success_avg = np.nanmean(expobj.SLMTargets_stims_dff[cell][success_stims], axis=0)
                failures_avg = np.nanmean(expobj.SLMTargets_stims_dff[cell][fail_stims], axis=0)
                axs[a, b].plot(success_avg, color='navy', linewidth=2, zorder=4)
                axs[a, b].plot(failures_avg, color='black', linewidth=2, zorder=4)
                axs[a, b].set_ylim([-0.1 * 100, 0.6 * 100])
                axs[a, b].text(0.98, 0.97,
                               'Success rate: %s' % ('{:,.1f}'.format(expobj.StimSuccessRate_SLMtargets[cell])),
                               verticalalignment='top', horizontalalignment='right',
                               transform=axs[a, b].transAxes, fontweight='bold',
                               color='black')
                axs[a, b].margins(0)
                axs[a, b].axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, color='mistyrose',
                                  zorder=0)

                counter += 1
            fig.suptitle((str(exp) + ' %s - %s targets' % ('- values: pct. dff', len(expobj.SLMTargets_stims_dff))))
            fig.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s_individual targets dFF.png' % (
            date, j[:-6], date, j))
            fig.show()

        # for x in range(expobj.SLMTargets_stims_dff[cell].shape[0]):
        #     response = expobj.responses_SLMtargets.loc[cell, expobj.]
        #     trace = expobj.SLMTargets_stims_dff[cell][x]
        #
        #     response = np.mean(trace[expobj.pre_stim_sec + expobj.stim_duration_frames + 1:
        #                                          expobj.pre_stim_sec + expobj.stim_duration_frames +
        #                                          expobj.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
        #     if response >= 0.1*100:
        #         responses_magnitudes_successes[cell].append(round(response, 2))
        #         response_traces_successes[cell] = np.vstack((trace, response_traces_successes[cell]))
        #         axs[a, b].plot(trace, color='skyblue', zorder=2, alpha=0.05)
        #     else:
        #         responses_magnitudes_failures[cell].append(round(response, 2))
        #         response_traces_failures[cell] = np.vstack((trace, response_traces_failures[cell]))
        #         axs[a, b].plot(trace, color='gray', zorder=3, alpha=0.05)
        # make plot for each individual cell

        #     success_plots = np.nanmean(response_traces_successes[cell][:-1], axis=0)
        #     failures_plots = np.nanmean(response_traces_failures[cell][:-1], axis=0)
        #     axs[a, b].plot(success_plots, color='navy', linewidth=2, zorder=4)
        #     axs[a, b].plot(failures_plots, color='black', linewidth=2, zorder=4)
        #     axs[a, b].axvspan(expobj.pre_stim_sec, expobj.pre_stim_sec + expobj.stim_duration_frames, color='mistyrose',
        #                       zorder=0)
        #     # ax.plot(response_traces_failures[cell][1:], color='black', zorder=1, alpha=0.1)
        #     # ax.plot(np.mean(expobj.SLMTargets_stims_raw[0], axis=0), color='black', zorder=1)
        #     axs[a, b].set_ylim([-0.2*100, 1.2*100])
        #     axs[a, b].text(0.98, 0.97, 'Success rate: %s' % ('{:,.2f}'.format(
        #         len(responses_magnitudes_successes[cell]) / (
        #                     len(responses_magnitudes_failures[cell]) + len(responses_magnitudes_successes[cell])))),
        #                    verticalalignment='top', horizontalalignment='right',
        #                    transform=axs[a, b].transAxes, fontweight='bold',
        #                    color='black')
        #     counter += 1
        # fig.suptitle((str(i) + ' %s - %s targets' % ('- % dff', len(expobj.SLMTargets_stims_dff))), y=0.995)
        # plt.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s.png' % (date, j[:-6], date, j))
        # fig.show()

    #     for trace in raw_traces_stims[cell]:
    #         # calculate dFF (noramlized to pre-stim) for each trace
    #         # axs[0].plot(trace, color='black', alpha=0.1)
    #         pre_stim_mean = np.mean(trace[0:expobj.pre_stim_sec])
    #         response_trace = (trace - pre_stim_mean)
    #         response_trace1 = response_trace / pre_stim_mean
    #         # if np.nanmax(response_trace) > 1e100 and np.nanmin(response_trace) < -1e100:
    #         #     print('\n%s' % np.nanmean(response_trace))
    #         #     print(np.nanmax(response_trace))
    #         #     print(np.nanmin(response_trace))
    #         std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #         response_trace2 = response_trace / std_pre
    #         measure = 'dF/F'
    #         to_plot = response_trace1
    #         # axs[1].plot(response_trace, color='green', alpha=0.1)
    #         # axs[2].plot(response_trace2, color='purple', alpha=0.1)
    #     # axs[2].axvspan(expobj.pre_stim_sec + expobj.stim_duration_frames, expobj.pre_stim_sec + expobj.stim_duration_frames + 1 + expobj.post_stim_response_frames_window, color='tomato')
    #         # response_trace = response_trace / std_pre
    #         # if dff_threshold:  # calculate dFF response for each stim trace
    #         #     response_trace = ((trace - pre_stim_mean)) #/ pre_stim_mean) * 100
    #         # else:  # calculate dF_stdF response for each stim trace
    #         #     pass
    #
    #         # calculate if the current trace beats the threshold for calculating reliability (note that this happens over a specific window just after the photostim)
    #         response = np.mean(to_plot[expobj.pre_stim_sec + expobj.stim_duration_frames + 1:
    #                                              expobj.pre_stim_sec + expobj.stim_duration_frames +
    #                                              expobj.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
    #
    #         # response_result = response / std_pre  # normalize the delta F above pre-stim mean using std of the pre-stim
    #         # response_trace = response_trace / std_pre
    #         if response >= 0.2:
    #             responses_magnitudes_successes[cell].append(round(response, 2))
    #             response_traces_successes[cell] = np.vstack((to_plot, response_traces_successes[cell]))
    #             axs[a, b].plot(to_plot, color='seagreen', zorder=3, alpha=0.1)
    #         else:
    #             responses_magnitudes_failures[cell].append(round(response, 2))
    #             response_traces_failures[cell] = np.vstack((to_plot, response_traces_failures[cell]))
    #             axs[a, b].plot(to_plot, color='gray', zorder=2, alpha=0.1)
    #
    #     # make plot for each individual cell
    #     success_plots = np.nanmean(response_traces_successes[cell][:-1], axis=0)
    #     failures_plots = np.nanmean(response_traces_failures[cell][:-1], axis=0)
    #     axs[a, b].plot(success_plots, color='darkgreen', linewidth=2, zorder=4)
    #     axs[a, b].plot(failures_plots, color='black', linewidth=2, zorder=4)
    #     axs[a, b].axvspan(expobj.pre_stim_sec, expobj.pre_stim_sec + expobj.stim_duration_frames, color='mistyrose',
    #                       zorder=0)
    #     # ax.plot(response_traces_failures[cell][1:], color='black', zorder=1, alpha=0.1)
    #     # ax.plot(np.mean(expobj.SLMTargets_stims_raw[0], axis=0), color='black', zorder=1)
    #     axs[a, b].set_ylim([-0.2, 1.2])
    #     axs[a, b].text(0.98, 0.97, 'Success rate: %s' % ('{:,.2f}'.format(len(responses_magnitudes_successes[cell]) / (len(responses_magnitudes_failures[cell]) + len(responses_magnitudes_successes[cell])))),
    #             verticalalignment='top', horizontalalignment='right',
    #             transform=axs[a, b].transAxes, fontweight='bold',
    #             color='black')
    #     counter += 1
    # fig.suptitle((str(i) + ' %s - %s targets' % (measure, len(raw_traces_stims))), y=0.995)
    # plt.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s.png' % (date, j[:-6], date, j))
    # fig.show()

"""# for i in range(len(response_traces_successes[cell][:-1])):
#     plt.plot(response_traces_successes[cell][:-1][i])
# plt.show()

# for i in responses_magnitudes_successes.keys():
#     print(len(responses_magnitudes_successes))
"""



# %% 2) BAR PLOT FOR PHOTOSTIM RESPONSE MAGNITUDE B/W PRE AND POST 4AP TRIALS - tODO plot delta (trace dFF) responses
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses['prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

post4ap_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    post4ap_response_magnitude.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_response_magnitude, post4ap_response_magnitude], paired=True,
                        colors=['black', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'post-4ap'], title='Avg. Response magnitude',
                        y_label='response magnitude')


# %% 3) BAR PLOT FOR PHOTOSTIM RESPONSE RELIABILITY B/W PRE AND POST 4AP TRIALS
pre4ap_reliability = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean reliability (>0.3 dF/stdF)'].values[0] for trial in i]
    pre4ap_reliability.append(np.mean(x))

post4ap_reliability = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean reliability (>0.3 dF/stdF)'].values[0] for trial in i]
    post4ap_reliability.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_reliability, post4ap_reliability], paired=True,
                        colors=['black', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'post-4ap'], title='Avg. Response Reliability',
                        y_label='% success rate of photostim')


# %% 4) plot responses of SLM TARGETS in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
# - with option to plot only successful or only failure stims!

# ## 3.1) PRE-4AP TRIALS
redo_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True  # avg only for each expobj
to_plot = 'successes'  # use for plotting either 'successes' stim responses or 'failures' stim responses

dffTraces = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.pre_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.pre_4ap_trials[0]
        # j = 0
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.1.0]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if redo_processing:
            aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                        plots=False, force_redo=False)
            expobj.save()


        # if not hasattr(expobj, 'traces_SLMtargets_successes_avg'):
        #     print('running .calculate_SLMTarget_SuccessStims method for expobj of %s, %s [1.1.1]' % (prep, trial))
        #     expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
        #     expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
        #     expobj.traces_SLMtargets_failures_avg = \
        #         expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_successes_avg[key] for key in
                            expobj.traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_failures_avg[key] for key in
                                        expobj.traces_SLMtargets_failures_avg.keys()])

        # prepare data for plotting
        y_label = '% dFF (normalized to prestim period)'
        x_label = 'Time (secs)'
        pre_stim = 0.25; pre_stim_fr = expobj.fps * pre_stim
        post_stim = 2.75; post_stim_fr = expobj.fps * post_stim
        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 3 frames as stimduration
                trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - run_pre4ap_trials stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - run_pre4ap_trials stims %s %s' % (to_plot, prep, trial)
        # make plot
        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj, stim_duration=stim_dur, y_lims=[0, 50],
                                                 pre_stim_sec=0.25, post_stim_sec=2.75, avg_only=avg_only, title=title,
                                                 y_label=y_label, x_label=x_label, fig=f, ax=ax, show=False)


        print('|- shape of dFF array: ', data_traces.shape, ' [1.1.3]')
        print('exp_obj prestim / post stim: %s, %s' % (expobj.pre_stim/expobj.fps - 1/expobj.fps, expobj.pre_stim/expobj.fps + stim_dur))
        dffTraces.append(d[1:3])

f.show()
allopticalResults.dffTraces = np.asarray(dffTraces)
allopticalResults.save()





# ## 3.2) POST-4AP TRIALS - IN SZ STIMS - EXCLUDE STIMS/CELLS INSIDE SZ BOUNDARY
run_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True
to_plot = 'failures'  # use for plotting either 'successes' stim responses or 'failures' stim responses

dffTraces_insz = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.post_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[0]
        # j = 0
        # prep = 'RL109'
        # trial = 't-016'
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.2.1]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if 'post' in expobj.metainfo['exptype']:
            if run_processing:
                aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                            plots=False, force_redo=False)
            expobj.save()

        #### use expobj.hits_SLMtargets for determining which photostim trials to use - setting this up to only plot successfull trials
        if not hasattr(expobj, 'insz_traces_SLMtargets_successes_avg'):
            stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            if len(stims_insz_idx) > 0:
                print('|- calculating stim success rates (insz) - %s stims [1.3.3]' % len(stims_insz_idx))
                expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_traces_SLMtargets_successes_avg, \
                expobj.insz_traces_SLMtargets_failures_avg = \
                    expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets,
                                                            stims_idx_l=stims_insz_idx,
                                                            exclude_stims_targets=expobj.slmtargets_szboundary_stim)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_successes_avg[key] for key in expobj.insz_traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_failures_avg[key] for key in expobj.insz_traces_SLMtargets_failures_avg.keys()])

        y_label = 'pct. dFF (normalized to prestim period)'

        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 5 frames as stimduration
                trace_ = np.append(trace_, trace[-expobj.post_stim:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - in sz stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - in sz stims %s %s' % (to_plot, prep, trial)

        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj,
                                                 stim_duration=stim_dur, y_lims=[0, 50], title=title, avg_only=avg_only,
                                                 pre_stim_sec=0.25, post_stim_sec=2.75,
                                                 y_label=y_label, x_label='Time (secs)', fig=f, ax=ax, show=False)

        print('|- shape of dFF array: ', data_traces.shape, ' [1.2.4]')
        dffTraces_insz.append(d)

f.show()
allopticalResults.dffTraces_outsz = np.asarray(dffTraces_insz)
allopticalResults.save()



# ## 3.3) POST-4AP TRIALS (OUT SZ STIMS)

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






# 3.3.1)
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