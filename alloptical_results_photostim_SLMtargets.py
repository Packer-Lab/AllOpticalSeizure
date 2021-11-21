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
import utils.funcs_pj as pj
import tifffile as tf
from skimage.transform import resize
from mpl_toolkits import mplot3d

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-17'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


# expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre e.1')  # PLACEHOLDER IMPORT OF EXPOBJ TO MAKE THE CODE WORK


# %%
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
# %% 8.2-dc) PLOT - absolute stim responses vs. TIME to seizure onset

"""todo for this analysis:
- average over targets for plot containing all exps
"""

# plotting of post_4ap zscore_stim_relative_to_sz onset
print(f"plotting averages from trials: {list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())}")

preps = np.unique([prep[:-6] for prep in allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys()])

exps = list(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())

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
            ax2.set_xlim(-300, 250)
            ax2.set_title(f"{key}")
            counter += 1

ax.set_xlim(-300, 250)
ax.set_xlabel('Time to closest seizure onset (secs)')
ax.set_ylabel('responses')

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




# %% 8.1-dc) PLOT - absolute stim responses vs. TIME to seizure onset

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




# %% 0) plot representative experiment plot for stim responses - showing pre4ap and post4ap

##  PRE4AP TRIAL
i = allopticalResults.post_4ap_trials[0]
j = 0
prep = i[j][:-6]
trial = i[j][-5:]
print('\nprogress @ ', prep, trial, ' [1.1.0]')
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

to_plot = 'dFstdF'

if to_plot == 'dFstdF':
    arr = np.asarray([i for i in expobj.targets_dfstdF_avg])
    y_label = 'dFstdF (normalized to prestim period)'
    y_lims = [-0.25,2.5]
elif to_plot == 'dFF':
    arr = np.asarray([i for i in expobj.targets_dff_avg])
    y_label = 'dFF (normalized to prestim period)'
    y_lims = [-15, 75]
aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=0.5, post_stim_sec=2.5,
                              title=(f'{prep} {trial} photostim targets'), figsize=[3,4], y_label=y_label,
                              x_label='Time', y_lims=y_lims)




##  POST4AP TRIAL
i = allopticalResults.pre_4ap_trials[0]
j = 0
prep = i[j][:-6]
trial = i[j][-5:]
print('\nprogress @ ', prep, trial, ' [1.1.0]')
expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)


# %% 1) BAR PLOT FOR PHOTOSTIM RESPONSE MAGNITUDE B/W PRE AND POST 4AP TRIALS
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
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

# %% 2) BAR PLOT FOR PHOTOSTIM RESPONSE RELIABILITY B/W PRE AND POST 4AP TRIALS
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

# %% 3) plot responses of SLM TARGETS in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
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


        if not hasattr(expobj, 'traces_SLMtargets_successes_avg'):
            print('running .calculate_SLMTarget_SuccessStims method for expobj of %s, %s [1.1.1]' % (prep, trial))
            expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
            expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
            expobj.traces_SLMtargets_failures_avg = \
                expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)

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
            title = '%s stims only, all exps. - avg. responses of photostim targets - pre4ap stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - pre4ap stims %s %s' % (to_plot, prep, trial)
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






# %% 3.3.1)
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


# %% 3.4) DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.post_4ap_trials + allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_out_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (outsz) [1.4.2] ***********')
                    # expobj.StimSuccessRate_SLMtargets_outsz, expobj.hits_SLMtargets_outsz, expobj.responses_SLMtargets_outsz, expobj.traces_SLMtargets_successes_outsz = \
                    #     expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stims_out_sz)
                    success_responses = expobj.hits_SLMtargets_outsz * expobj.responses_SLMtargets_outsz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response outsz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())

                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_in_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (insz) [1.4.3] ***********')
                    # expobj.StimSuccessRate_SLMtargets_insz, expobj.hits_SLMtargets_insz, expobj.responses_SLMtargets_insz, expobj.traces_SLMtargets_successes_insz = \
                    #     expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stims_in_sz)

                    success_responses = expobj.hits_SLMtargets_insz * expobj.responses_SLMtargets_insz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response insz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())


            elif 'pre' in expobj.metainfo['exptype']:
                seizure_filter = False
                print('\n Calculating stim success rates and response magnitudes [1.4.4] ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
                    expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stim_start_frames)

                success_responses = expobj.hits_SLMtargets * expobj.responses_SLMtargets
                success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response (hits, all targets)'] = success_responses.mean()
                print(success_responses.mean())

            expobj.save()
    allopticalResults.save()


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


# %% 3.5) COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets_outsz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_outsz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response outsz (failures, all targets)'] = failure_responses.mean()

                inverse = (expobj.hits_SLMtargets_insz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_insz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response insz (failures, all targets)'] = failure_responses.mean()

            elif 'pre' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response (failures, all targets)'] = failure_responses.mean()
    allopticalResults.save()


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










# %% 4) plot peri-photostim avg traces for all trials analyzed to make sure they look alright -- plot as little postage stamps

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
                print('\n Calculating stim success rates and response magnitudes [4.1] ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=10,
                                                              stims_to_use=expobj.stim_start_frames)
                expobj.save()

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

# %% 5.0-main) TODO collect SLM targets responses for stims dynamically over time

"""# plot the target photostim responses for individual targets for each stim over the course of the trial
#    (normalize to each target's overall mean response) and plot over the timecourse of the trial

# # ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
# for key in ls(allopticalResults.trial_maps['post'].keys())[-5:]:
#     for j in range(len(allopticalResults.trial_maps['post'][key])):
#         # import expobj
#         expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))


# ls = ['RL108 t-013', 'RL109 t-021', 'RL109 t-016']
# # ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
# for key in ls(allopticalResults.trial_maps['post'].keys())[-5:]:
#     for j in range(len(allopticalResults.trial_maps['post'][key])):
#         # import expobj
#         expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j), do_processing=True)
"""


# %% 5.1) collect SLM targets responses for stims dynamically over time - APPROACH #1 - CALCULATING RESPONSE MAGNITUDE AT EACH STIM PER TARGET
key = 'e'
j = 0
exp = 'post'
expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.{j}")

SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
target_colors = pj.make_random_color_array(len(SLMtarget_ids))

# --- plot with mean FOV fluorescence signal
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
fig, axs[0] = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', figsize=[20, 3], show=False,
                                         fig=fig, ax=axs[0])
ax2 = axs[0].twinx()

## calculate and plot the response magnitude for each target at each stim;
#   where response magnitude is classified as response of each target at a particular stim relative to the mean response from the whole trial
for target in expobj.responses_SLMtargets.index:
    mean_response = np.mean(expobj.responses_SLMtargets.iloc[target, :])
    # print(mean_response)
    for i in expobj.responses_SLMtargets.columns:
        response = expobj.responses_SLMtargets.iloc[target, i] - mean_response
        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
        ax2.scatter(x=expobj.stim_start_frames[i] + rand, y=response, color=target_colors[target], alpha=0.70, s=15, zorder=4)
        ax2.axhline(y=0)
        ax2.set_ylabel('Response mag. (relative to mean)')
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
fig, axs[1] = aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='Time', show=False, fig=fig, ax=axs[1])
ax2.margins(x=0)
fig.suptitle(f"Photostim responses - {exp}-4ap {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}")
fig.show()



# %% 5.2) collect SLM targets responses for stims dynamically over time - APPROACH #2 - USING Z-SCORED PHOTOSTIM RESPONSES

print(f"---------------------------------------------------------")
print(f"plotting zscored photostim responses over the whole trial")
print(f"---------------------------------------------------------")


### PRE 4AP
trials = list(allopticalResults.trial_maps['pre'].keys())
fig, axs = plt.subplots(nrows=len(trials) * 2, ncols=1, figsize=[20, 6 * len(trials)])
counter = 0
for expprep in list(allopticalResults.stim_responses_zscores.keys())[:3]:
    for trials_comparisons in allopticalResults.stim_responses_zscores[expprep]:
        pre4ap_trial = trials_comparisons[:5]
        post4ap_trial = trials_comparisons[-5:]

        # PRE 4AP STUFF
        if f"{expprep} {pre4ap_trial}" in pj.flattenOnce(allopticalResults.pre_4ap_trials):
            pre4ap_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['pre-4ap']

            print(f"working on expobj: {expprep} {pre4ap_trial}, counter @ {counter}")
            expobj, experiment = aoutils.import_expobj(prep=expprep, trial=pre4ap_trial)

            SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
            target_colors = pj.make_random_color_array(len(SLMtarget_ids))
            # --- plot with mean FOV fluorescence signal
            # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
            ax = axs[counter]
            fig, ax = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', show=False, fig=fig, ax=ax)
            ax2 = ax.twinx()
            ## retrieve the appropriate zscored database - pre4ap stims
            targets = [x for x in list(pre4ap_df.columns) if type(x) == str and '_z' in x]
            for target in targets:
                for stim_idx in pre4ap_df.index[:-2]:
                    # if i == 'pre':
                    #     stim_idx = expobj.stim_start_frames.index(stim_idx)  # MINOR BUG: appears that for some reason the stim_idx of the allopticalResults.stim_responses_zscores for pre-4ap are actually the frames themselves
                    response = pre4ap_df.loc[stim_idx, target]
                    rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                    ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response, color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

            ax2.axhline(y=0)
            ax2.set_ylabel('Response mag. (zscored to pre4ap)')
            ax2.margins(x=0)

            ax3 = axs[counter + 1]
            ax3_2 = ax3.twinx()
            fig, ax3, ax3_2 = aoplot.plot_lfp_stims(expobj, x_axis='Time', show=False, fig=fig, ax=ax3, ax2=ax3_2)

            counter += 2
            print(f"|- finished on expobj: {expprep} {pre4ap_trial}, counter @ {counter}\n")

fig.suptitle(f"Photostim responses - pre-4ap", y=0.99)
save_path_full = f"{save_path_prefix}/pre4ap_indivtrial_zscore_responses.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()



### POST 4AP
trials_to_plot = pj.flattenOnce(allopticalResults.post_4ap_trials)
fig, axs = plt.subplots(nrows=len(trials_to_plot) * 2, ncols=1, figsize=[20, 6 * len(trials_to_plot)])
post4ap_trials_stimresponses_zscores = list(allopticalResults.stim_responses_zscores.keys())
counter = 0
for expprep in post4ap_trials_stimresponses_zscores:
    for trials_comparisons in allopticalResults.stim_responses_zscores[expprep]:
        if len(allopticalResults.stim_responses_zscores[expprep][trials_comparisons].keys()) > 2:  ## make sure that there are keys containing data for post 4ap and in sz
            pre4ap_trial = trials_comparisons[:5]
            post4ap_trial = trials_comparisons[-5:]

            # POST 4AP STUFF
            if f"{expprep} {post4ap_trial}" in trials_to_plot:
                post4ap_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['post-4ap']

                insz_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['in sz']


                print(f"working on expobj: {expprep} {post4ap_trial}, counter @ {counter}")
                expobj, experiment = aoutils.import_expobj(prep=expprep, trial=post4ap_trial)

                SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
                target_colors = pj.make_random_color_array(len(SLMtarget_ids))
                # --- plot with mean FOV fluorescence signal
                # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
                ax = axs[counter]
                fig, ax = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', show=False, fig=fig, ax=ax)
                ax.margins(x=0)

                ax2 = ax.twinx()
                ## retrieve the appropriate zscored database - post4ap (outsz) stims
                targets = [x for x in list(post4ap_df.columns) if type(x) == str and '_z' in x]
                assert len(targets) == len(SLMtarget_ids), print('mismatch in SLMtargets_ids and targets post4ap out sz')
                for target in targets:
                    for stim_idx in post4ap_df.index[:-2]:
                        response = post4ap_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                        assert not np.isnan(response)
                        ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response, color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)


                ## retrieve the appropriate zscored database - insz stims
                targets = [x for x in list(insz_df.columns) if type(x) == str]
                assert len(targets) == len(SLMtarget_ids), print('mismatch in SLMtargets_ids and targets in sz')
                for target in targets:
                    for stim_idx in insz_df.index:
                        response = insz_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                        if not np.isnan(response):
                            ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response, color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

                ax2.axhline(y=0)
                ax2.set_ylabel('Response mag. (zscored to pre4ap)')
                ax2.margins(x=0)

                ax3 = axs[counter+1]
                ax3_2 = ax3.twinx()
                fig, ax3, ax3_2 = aoplot.plot_lfp_stims(expobj, x_axis='Time', show=False, fig=fig, ax=ax3, ax2=ax3_2)

                counter += 2
                print(f"|- finished on expobj: {expprep} {post4ap_trial}, counter @ {counter}\n")

fig.suptitle(f"Photostim responses - post-4ap", y=0.99)
save_path_full = f"{save_path_prefix}/post4ap_indivtrial_zscore_responses.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()


# %% 6.0-main-dc) TODO collect targets responses for stims vs. distance (starting with old code)- low priority right now

key = 'e'
j = 0
exp = 'post'
expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.{j}")



# plot response magnitude vs. distance
for i in range(len(expobj.stim_times)):
    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
    s2pcells = expobj.cells_sz_stim[expobj.stim_start_frames[i]]
    target_coord = expobj.target_coords_all[target]
    min_distance = 1000
    for j in range(len(s2pcells)):
        dist = pj.calc_distance_2points(target_coord, tuple(expobj.stat[j]['med']))  # distance in pixels
        if dist < min_distance:
            min_distance = dist

fig1, ax1 = plt.subplots(figsize=[5, 5])
responses = []
distance_to_sz = []
responses_ = []
distance_to_sz_ = []
for target in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[target])
    target_coord = expobj.target_coords_all[target]
    # print(mean_response)

    # calculate response magnitude at each stim time for selected target
    for i in range(len(expobj.stim_times)):
        # the response magnitude of the current SLM target at the current stim time (relative to the mean of the responses of the target over this trial)
        response = expobj.responses_SLMtargets[target][i] / mean_response  # changed to division by mean response instead of substracting
        min_distance = pj.calc_distance_2points((0, 0), (expobj.frame_x,
                                                         expobj.frame_y))  # maximum distance possible between two points within the FOV, used as the starting point when the sz has not invaded FOV yet

        if hasattr(expobj, 'cells_sz_stim') and expobj.stim_start_frames[i] in list(expobj.cells_sz_stim.keys()):  # calculate distance to sz only for stims where cell locations in or out of sz boundary are defined in the seizures
            if expobj.stim_start_frames[i] in expobj.stims_in_sz:
                # collect cells from this stim that are in sz
                s2pcells_sz = expobj.cells_sz_stim[expobj.stim_start_frames[i]]

                # classify the SLM target as in or out of sz, if out then continue with mesauring distance to seizure wavefront,
                # if in sz then assign negative value for distance to sz wavefront
                sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path, expobj.metainfo['trial'], expobj.stim_start_frames[i])

                in_sz_bool = expobj._InOutSz(cell_med=[target_coord[1], target_coord[0]],
                                             sz_border_path=sz_border_path)

                if expobj.stim_start_frames[i] in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True
                    in_sz_bool = not in_sz_bool

                if in_sz_bool is True:
                    min_distance = -1

                else:
                    ## working on add feature for edgecolor of scatter plot based on calculated distance to seizure
                    ## -- thinking about doing this as comparing distances between all targets and all suite2p ROIs,
                    #     and the shortest distance that is found for each SLM target is that target's distance to seizure wavefront
                    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
                    if len(s2pcells_sz) > 0:
                        for j in range(len(s2pcells_sz)):
                            s2p_idx = expobj.cell_id.index(s2pcells_sz[j])
                            dist = pj.calc_distance_2points(target_coord, tuple(
                                [expobj.stat[s2p_idx]['med'][1], expobj.stat[s2p_idx]['med'][0]]))  # distance in pixels
                            if dist < min_distance:
                                min_distance = dist

        if min_distance > 600:
            distance_to_sz_.append(min_distance + np.random.randint(-10, 10, 1)[0] - 165)
            responses_.append(response)
        elif min_distance > 0:
            distance_to_sz.append(min_distance)
            responses.append(response)

# calculate linear regression line
ax1.plot(range(int(min(distance_to_sz)), int(max(distance_to_sz))), np.poly1d(np.polyfit(distance_to_sz, responses, 1))(range(int(min(distance_to_sz)), int(max(distance_to_sz)))),
         color='black')

ax1.scatter(x=distance_to_sz, y=responses, color='cornflowerblue',
            alpha=0.5, s=16, zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.scatter(x=distance_to_sz_, y=responses_, color='firebrick',
            alpha=0.5, s=16, zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.set_xlabel('distance to seizure front (pixels)')
ax1.set_ylabel('response magnitude')
ax1.set_title('')
fig1.show()






# %% 7.0-main) avg responses in space around photostim targets - pre vs. post4ap

# %% 8.0-dc) PLOT - zscore of stim responses vs. TIME to seizure onset

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




# %% 9.0-dc) zscore of stim responses vs. TIME to seizure onset - original code for single experiments
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
    closest_sz_onset = pj.findClosest(ls=expobj.seizure_lfp_onsets, input=stim_frame)[0]
    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
    stims_relative_sz.append(round(time_diff, 3))

cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x=post_4ap_df_zscore_stim_relative_to_sz.index, y=post_4ap_df_zscore_stim_relative_to_sz['avg'])
fig.show()