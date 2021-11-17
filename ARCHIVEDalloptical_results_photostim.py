# ARCHIVED ON: NOV 15 2021

# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
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

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-11'
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

"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""
"""# sys.exit()
###########
"""


# %% 7.0-main) TODO collect targets responses for stims dynamically over time (starting with old code)

# plot the target photostim responses for individual targets for each stim over the course of the trial
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



# %% 7.0-dc)
## APPROACH #1 - CALCULATING RESPONSE MAGNITUDE AT EACH STIM PER TARGET
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



# %% 7.0-dc)
# APPROACH #2 - USING Z-SCORED PHOTOSTIM RESPONSES

print(f"---------------------------------------------------------")
print(f"plotting zscored photostim responses over the whole trial")
print(f"---------------------------------------------------------")

### PRE 4AP
trials = list(allopticalResults.trial_maps['pre'].keys())
fig, axs = plt.subplots(nrows=len(trials) * 2, ncols=1, figsize=[20, 6 * len(trials)])
counter = 0
for expprep in list(allopticalResults.stim_responses_zscores.keys()):
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

            ax2.margins(x=0)

            ax3 = axs[counter + 1]
            ax3_2 = ax3.twinx()
            fig, ax3, ax3_2 = aoplot.plot_lfp_stims(expobj, x_axis='Time', show=False, fig=fig, ax=ax3, ax2=ax3_2)

            counter += 2
            print(f"|- finished on expobj: {expprep} {pre4ap_trial}, counter @ {counter}\n")

fig.suptitle(f"Photostim responses - pre-4ap", y=0.99)
fig.show()
# %%


### POST 4AP
trials = list(allopticalResults.trial_maps['post'].keys())
fig, axs = plt.subplots(nrows=len(trials) * 2, ncols=1, figsize=[20, 6 * len(trials)])
counter = 0
for expprep in list(allopticalResults.stim_responses_zscores.keys()):
    for trials_comparisons in allopticalResults.stim_responses_zscores[expprep]:
        if len(allopticalResults.stim_responses_zscores[expprep][trials_comparisons].keys()) > 2:  ## make sure that there are keys containing data for post 4ap and in sz
            pre4ap_trial = trials_comparisons[:5]
            post4ap_trial = trials_comparisons[-5:]

            # POST 4AP STUFF
            if f"{expprep} {post4ap_trial}" in pj.flattenOnce(allopticalResults.post_4ap_trials):
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
                for target in targets:
                    for stim_idx in post4ap_df.index[:-2]:
                        response = post4ap_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                        ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response, color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

                ax2.margins(x=0)


                ## retrieve the appropriate zscored database - insz stims
                targets = [x for x in list(insz_df.columns) if type(x) == str]
                for target in targets:
                    for stim_idx in insz_df.index:
                        response = insz_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[0] #* 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
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
fig.show()


# %% 5.1.1-dc) PLOT - dF/F of significant pos. and neg. responders that were derived from dF/stdF method
print('\n----------------------------------------------------------------')
print('plotting dFF for significant cells ')
print('----------------------------------------------------------------')

expobj.sig_cells = [expobj.s2p_nontargets[i] for i, x in enumerate(expobj.sig_units) if x]
expobj.pos_sig_cells = [expobj.sig_cells[i] for i in np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
expobj.neg_sig_cells = [expobj.sig_cells[i] for i in np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
# plot peristim avg dFF of pos_sig_cells
selection = [expobj.s2p_nontargets.index(i) for i in expobj.pos_sig_cells]
x = expobj.dff_traces_avg[selection]
y_label = 'pct. dFF (normalized to prestim period)'
f, ax[0, 0], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title='positive sig. responders', y_label=y_label, fig=f, ax=ax[0, 0], show=False,
                              x_label=None, y_lims=[-50, 200])

# plot peristim avg dFF of neg_sig_cells
selection = [expobj.s2p_nontargets.index(i) for i in expobj.neg_sig_cells]
x = expobj.dff_traces_avg[selection]
y_label = None
f, ax[0, 1], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title='negative sig. responders', y_label=None, fig=f, ax=ax[0, 1], show=False,
                              x_label=None, y_lims=[-50, 200])

# plot peristim avg dFstdF of pos_sig_cells
selection = [expobj.s2p_nontargets.index(i) for i in expobj.pos_sig_cells]
x = expobj.dfstdF_traces_avg[selection]
y_label = 'dF/stdF'
f, ax[1, 0], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title=None, y_label=y_label, fig=f, ax=ax[1, 0], show=False,
                              x_label='Time (seconds) ', y_lims=[-1, 1])

# plot peristim avg dFstdF of neg_sig_cells
selection = [expobj.s2p_nontargets.index(i) for i in expobj.neg_sig_cells]
x = expobj.dfstdF_traces_avg[selection]
y_label = None
f, ax[1, 1], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title=None, y_label=y_label, fig=f, ax=ax[1, 1], show=False,
                              x_label='Time (seconds) ', y_lims=[-1, 1])
f.show()



# %% 5.1.2-dc) PLOT - creating large figures collating multiple plots describing responses of non targets to photostim for individual expobj's -- collecting code in aoutils.fig_non_targets_responses()
plot_subset = False

if plot_subset:
    selection = np.random.randint(0, expobj.dff_traces_avg.shape[0], 100)
else:
    selection = np.arange(expobj.dff_traces_avg.shape[0])

#### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
f = plt.figure(figsize=[30, 10])
gs = f.add_gridspec(2, 9)

# %% 5.1.2.1-dc) PLOT OF PERI-STIM AVG TRACES FOR ALL SIGNIFICANT AND NON-SIGNIFICANT RESPONDERS - also breaking down positive and negative responders

# PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
a1 = f.add_subplot(gs[0, 0:2])
x = expobj.dff_traces_avg[selection]
y_label = 'pct. dFF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=4,
                              title=None, y_label=y_label, fig=f, ax=a1, show=False,
                              x_label='Time (seconds)', y_lims=[-50, 200])
# PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
a2 = f.add_subplot(gs[0, 2:4])
x = expobj.dfstdF_traces_avg[selection]
y_label = 'dFstdF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=4,
                              title=None, y_label=y_label, fig=f, ax=a2, show=False,
                              x_label='Time (seconds)', y_lims=[-1, 3])
# PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/f
a3 = f.add_subplot(gs[0, 4:6])
vmin = -1
vmax = 1
aoplot.plot_traces_heatmap(expobj.dfstdF_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=int(1 * expobj.fps),
                           stim_off=int(1 * expobj.fps + expobj.stim_duration_frames - 1), xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                           title='dF/F heatmap for all nontargets', x_label='Time', cbar=True,
                           fig=f, ax=a3, show=False)
# PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/stdf
a4 = f.add_subplot(gs[0, -3:-1])
vmin = -100
vmax = 100
aoplot.plot_traces_heatmap(expobj.dff_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=int(1 * expobj.fps),
                           stim_off=int(1 * expobj.fps + expobj.stim_duration_frames - 1), xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                           title='dF/stdF heatmap for all nontargets', x_label='Time', cbar=True,
                           fig=f, ax=a4, show=False)

# plot PERI-STIM AVG TRACES of sig nontargets
a10 = f.add_subplot(gs[1, 0:2])
x = expobj.dfstdF_traces_avg[expobj.sig_units]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a10, show=False,
                              title='significant responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of nonsig nontargets
a11 = f.add_subplot(gs[1, 2:4])
x = expobj.dfstdF_traces_avg[~expobj.sig_units]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a11, show=False,
                              title='non-significant responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of sig. positive responders
a12 = f.add_subplot(gs[1, 4:6])
x = expobj.dfstdF_traces_avg[expobj.sig_units][
    np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a12, show=False,
                              title='positive signif. responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of sig. negative responders
a13 = f.add_subplot(gs[1, -3:-1])
x = expobj.dfstdF_traces_avg[expobj.sig_units][
    np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a13, show=False,
                              title='negative signif. responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# %% 5.1.2.2-dc) PLOT - quantifying responses of non targets to photostim
# bar plot of avg post stim response quantified between responders and non-responders
a04 = f.add_subplot(gs[0, -1])
sig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)
nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
data = np.asarray([sig_responders_avgresponse, nonsig_responders_avgresponse])
pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'gray'], y_label='avg dF/stdF', bar=False,
                        text_list=['%s pct' % (np.round((len(sig_responders_avgresponse)/expobj.post_array_responses.shape[0]), 2) * 100),
                                   '%s pct' % (np.round((len(nonsig_responders_avgresponse)/expobj.post_array_responses.shape[0]), 2) * 100)],
                        text_y_pos=1.43, text_shift=1.7, x_tick_labels=['significant', 'non-significant'], expand_size_y=1.5, expand_size_x=0.6,
                        fig=f, ax=a04, show=False)


# bar plot of avg post stim response quantified between responders and non-responders
a14 = f.add_subplot(gs[1, -1])
possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
data = np.asarray([possig_responders_avgresponse, negsig_responders_avgresponse, nonsig_responders_avgresponse])
pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'blue', 'gray'], y_label='avg dF/stdF', bar=False,
                        text_list=['%s pct' % (np.round((len(possig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1)),
                                   '%s pct' % (np.round((len(negsig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1)),
                                   '%s pct' % (np.round((len(nonsig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1))],
                        text_y_pos=1.43, text_shift=1.2, x_tick_labels=['pos. significant', 'neg. significant', 'non-significant'], expand_size_y=1.5, expand_size_x=0.5,
                        fig=f, ax=a14, show=False)

f.suptitle(
    ('%s %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])))
f.show()






# %% 5.2.1) PLOT - scatter plot 1) response magnitude vs. prestim std F, and 2) response magnitude vs. prestim mean F
## TODO check if these plots are coming out okay...

# 5.2.1.1) scatter plot response magnitude vs. prestim std F
ls = ['post']
for i in ls:
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
    counter = 0

    j = 0
    for key in list(allopticalResults.trial_maps[i].keys()):

        expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')  # import expobj

        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)

        posunits_prestdF = np.mean(np.std(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0], :, :], axis=2), axis=1)
        negunits_prestdF = np.mean(np.std(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0], :, :], axis=2), axis=1)
        nonsigunits_prestdF = np.mean(np.std(expobj.raw_traces[~expobj.sig_units, :, :], axis=2), axis=1)

        assert len(possig_responders_avgresponse) == len(posunits_prestdF)
        assert len(negsig_responders_avgresponse) == len(negunits_prestdF)
        assert len(nonsig_responders_avgresponse) == len(nonsigunits_prestdF)

        # fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
        ax = axs[counter // ncols, counter % ncols]
        ax.scatter(x = nonsigunits_prestdF, y = nonsig_responders_avgresponse, color='gray', alpha=0.10, label='non sig.', s=65, edgecolors='none', zorder=0)
        ax.scatter(x = negunits_prestdF, y = negsig_responders_avgresponse, color='red', alpha=0.10, label='sig. neg.', s=65, edgecolors='none', zorder=1)
        ax.scatter(x = posunits_prestdF, y = possig_responders_avgresponse, color='green', alpha=0.10, label='sig. pos.', s=65, edgecolors='none', zorder=2)
        ax.set_title(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ")
        # fig.show()

        counter += 1
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Avg. prestim std F')
    axs[0, 0].set_ylabel('Avg. mag (dF/stdF)')
    fig.tight_layout()
    fig.suptitle(f'All exps. prestim std F vs. response mag (dF/stdF) distribution - {i}4ap', y = 0.98)
    save_path = save_path_prefix + f"/scatter prestim std F vs. plot response magnitude - {i}4ap.png"
    plt.savefig(save_path)
    fig.show()

# 5.2.1.2) scatter plot response magnitude vs. prestim mean F
ls = ['pre', 'post']
for i in ls:
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
    counter = 0

    j = 0
    for key in list(allopticalResults.trial_maps[i].keys()):

        expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')  # import expobj

        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)

        posunits_prestdF = np.mean(np.mean(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0], :, :], axis=2), axis=1)
        negunits_prestdF = np.mean(np.mean(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0], :, :], axis=2), axis=1)
        nonsigunits_prestdF = np.mean(np.mean(expobj.raw_traces[~expobj.sig_units, :, :], axis=2), axis=1)

        assert len(possig_responders_avgresponse) == len(posunits_prestdF)
        assert len(negsig_responders_avgresponse) == len(negunits_prestdF)
        assert len(nonsig_responders_avgresponse) == len(nonsigunits_prestdF)

        # fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
        ax = axs[counter // ncols, counter % ncols]
        ax.scatter(x = nonsigunits_prestdF, y = nonsig_responders_avgresponse, color='gray', alpha=0.10, label='non sig.', s=65, edgecolors='none', zorder = 0)
        ax.scatter(x = negunits_prestdF, y = negsig_responders_avgresponse, color='red', alpha=0.10, label='sig. neg.', s=65, edgecolors='none', zorder = 1)
        ax.scatter(x = posunits_prestdF, y = possig_responders_avgresponse, color='green', alpha=0.10, label='sig. pos.', s=65, edgecolors='none', zorder = 2)
        ax.set_title(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ")
        # fig.show()

        counter += 1
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Avg. prestim mean F')
    axs[0, 0].set_ylabel('Avg. mag (dF/stdF)')
    fig.tight_layout()
    fig.suptitle(f'All exps. prestim mean F vs. response mag (dF/stdF) distribution - {i}4ap')
    save_path = save_path_prefix + f"/scatter plot prestim mean F vs. response magnitude - {i}4ap.png"
    plt.savefig(save_path)
    fig.show()




# %% 5.2.2) PLOT - measuring avg raw pre-stim stdF for all non-targets - pre4ap vs. post4ap histogram comparison
ncols = 3
nrows = 4
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0

for key in list(allopticalResults.trial_maps['pre'].keys()):
    j = 0  # get just the first trial from the allopticalResults.trial_maps

    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, j))  # import expobj
    allunits_prestdF_pre4ap_ = np.mean(np.std(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)


    expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))  # import expobj
    allunits_prestdF_post4ap_ = np.mean(np.std(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)

    # plot the histogram
    ax = axs[counter // ncols, counter % ncols]
    fig, ax = pj.plot_hist_density([allunits_prestdF_pre4ap_, allunits_prestdF_post4ap_], x_label=None, legend_labels=['pre4ap', 'post4ap'],
                                   title=f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ", show_legend=False,
                                   fill_color=['gray', 'purple'], num_bins=100, fig=fig, ax=ax, show=False, shrink_text=0.7,
                                   figsize=(4, 5))
    counter += 1
axs[0, 0].legend()
axs[0, 0].set_ylabel('density')
axs[0, 0].set_xlabel('Avg. prestim std F')
fig.tight_layout()
fig.suptitle('All exps. prestim std F distribution - pre vs. post4ap')
save_path = save_path_prefix + f"/All exps. prestim std F distribution - pre vs. post4ap.png"
plt.savefig(save_path)
fig.show()


# 5.2.2.1) PLOT - measuring avg raw pre-stim stdF for all non-targets - pre4ap only
# key = 'h'; j = 0

sig_units_prestdF_pre4ap = []
nonsig_units_prestdF_pre4ap = []
allunits_prestdF_pre4ap = []

ncols = 3
nrows = 4
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0


for key in list(allopticalResults.trial_maps['pre'].keys()):
    for j in range(len(allopticalResults.trial_maps['pre'][key])):
        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, j))

        # get the std of pre_stim period for each photostim trial from significant and non-significant responders, averaged over all trials for each cell individually
        sig_units_prestdF_pre4ap_ = np.mean(np.std(expobj.raw_traces[expobj.sig_units, :, expobj.pre_stim_frames_test], axis=2), axis=1)
        nonsig_units_prestdF_pre4ap_ = np.mean(np.std(expobj.raw_traces[~expobj.sig_units, :, expobj.pre_stim_frames_test], axis=2), axis=1)

        sig_units_prestdF_pre4ap.append(sig_units_prestdF_pre4ap_)
        nonsig_units_prestdF_pre4ap.append(nonsig_units_prestdF_pre4ap_)

        allunits_prestdF_pre4ap_ = np.mean(np.std(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)
        allunits_prestdF_pre4ap.append(allunits_prestdF_pre4ap_)

        # plot the histogram
        ax = axs[counter // ncols, counter % ncols]
        fig, ax = pj.plot_hist_density([allunits_prestdF_pre4ap_], x_label=None, title=f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ",
                                       fill_color=['gray'], num_bins=100, fig=fig, ax=ax, show=False, shrink_text=0.7, figsize=(4, 5))
        counter += 1

axs[0, 0].set_ylabel('density')
axs[0, 0].set_xlabel('prestim std F')
title = 'All exps. prestim std F distribution - pre4ap only'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()




# 5.2.2.2) PLOT - measuring avg raw pre-stim stdF for all non-targets - post4ap trials
sig_units_prestdF_post4ap = []
nonsig_units_prestdF_post4ap = []
allunits_prestdF_post4ap = []

ncols = 3
nrows = 4
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0

# ls = pj.flattenx1(allopticalResults.post_4ap_trials)
for key in list(allopticalResults.trial_maps['post'].keys()):
    for j in range(len(allopticalResults.trial_maps['post'][key])):
        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))
        # compare std of significant and nonsignificant units
        sig_units_prestdF_post4ap_ = np.mean(
            np.std(expobj.raw_traces[expobj.sig_units, :, expobj.pre_stim_frames_test], axis=2), axis=1)
        nonsig_units_prestdF_post4ap_ = np.mean(
            np.std(expobj.raw_traces[~expobj.sig_units, :, expobj.pre_stim_frames_test], axis=2), axis=1)

        sig_units_prestdF_post4ap.append(sig_units_prestdF_post4ap_)
        nonsig_units_prestdF_post4ap.append(nonsig_units_prestdF_post4ap_)

        allunits_prestdF_post4ap_ = np.mean(np.std(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)
        allunits_prestdF_post4ap.append(allunits_prestdF_post4ap_)

        # plot the histogram
        ax = axs[counter // ncols, counter % ncols]
        fig, ax = pj.plot_hist_density([allunits_prestdF_post4ap_], x_label=None, y_label=None,
                                       title=f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ",
                                       fill_color=['purple'], num_bins=100, fig=fig, ax=ax, show=False, shrink_text=0.7,
                                       figsize=(4, 5))
        counter += 1

axs[0, 0].set_ylabel('density')
axs[0, 0].set_xlabel('prestim std F')
title = 'All exps. prestim std F distribution - post4ap only'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()






# %% 5.2.3) PLOT - measuring avg. raw prestim F - do post4ap cells have a lower avg. raw prestim F?

ncols = 3
nrows = 4
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0

for key in list(allopticalResults.trial_maps['pre'].keys()):
    j = 0  # get just the first trial from the allopticalResults.trial_maps

    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, j))  # import expobj
    raw_meanprestim_pre4ap = np.mean(np.mean(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)  # collect mean prestim for raw traces avg over trials - pre4ap

    expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))  # import expobj
    raw_meanprestim_post4ap = np.mean(np.mean(expobj.raw_traces[:, :, expobj.pre_stim_frames_test], axis=2), axis=1)  # collect mean prestim for raw traces avg over trials - post4ap

    # plot the histogram
    ax = axs[counter // ncols, counter % ncols]
    fig, ax = pj.plot_hist_density([raw_meanprestim_pre4ap, raw_meanprestim_post4ap], x_label=None, y_label=None,
                                   legend_labels=['pre4ap', 'post4ap'], title=f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ",
                                   show_legend=False, fill_color=['gray', 'purple'], num_bins=100, fig=fig, ax=ax, show=False,
                                   shrink_text=0.7, figsize=(4, 5))
    counter += 1
axs[0, 0].legend()
axs[0, 0].set_ylabel('density')
axs[0, 0].set_xlabel('Avg. prestim F')
title = 'All exps. prestim mean F distribution - pre vs. post4ap'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()






# %% 5.3.1) PLOT - bar plot average # of significant responders (+ve and -ve) for pre vs. post 4ap
data=[]
cols = ['pre4ap_pos', 'post4ap_pos']
for col in cols:
    data.append(list(allopticalResults.num_sig_responders_df.loc[:, col]))

cols = ['pre4ap_neg', 'post4ap_neg']
for col in cols:
    data.append(list(allopticalResults.num_sig_responders_df.loc[:, col]))


experiments = ['RL108', 'RL109', 'PS05', 'PS07', 'PS06', 'PS11']
pre4ap_pos = []
pre4ap_neg = []
post4ap_pos = []
post4ap_neg = []

for exp in experiments:
    rows = []
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            rows.append(row)
    x = allopticalResults.num_sig_responders_df.iloc[rows, :].mean(axis=0)
    pre4ap_pos.append(round(x[0], 1))
    pre4ap_neg.append(round(x[1], 1))
    post4ap_pos.append(round(x[2], 1))
    post4ap_neg.append(round(x[3], 1))


fig, axs = plt.subplots(ncols=2, nrows=1)
data = [pre4ap_pos, post4ap_pos]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap_pos', 'post4ap_pos'], colors=['lightgreen', 'forestgreen'],
                        bar=True, paired=True, expand_size_x=0.6, expand_size_y=1.3, title='# of Positive responders',
                        y_label='# of sig. responders', ax = axs[0], fig=fig, show=False)

data = [pre4ap_neg, post4ap_neg]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap_neg', 'post4ap_neg'], colors=['skyblue', 'steelblue'],
                        bar=True, paired=True, expand_size_x=0.6, expand_size_y=1.3, title='# of Negative responders',
                        y_label='# of sig. responders', ax=axs[1], fig=fig, show=False)
title = 'number of pos and neg responders pre vs. post4ap'
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()


# %% 5.3.2) PLOT - peri-stim average response stim graph for positive and negative followers
"""# - make one graph per comparison for now... then can figure out how to average things out later."""

# experiments = ['RL108t', 'RL109t', 'PS05t', 'PS07t', 'PS06t', 'PS11t']
experiments = ['RL109t', 'PS05t', 'PS07t', 'PS06t', 'PS11t']  # 'RL108t' already run successfully (RL109t008 vs. t021 had an issue)
pre4ap_pos = []
pre4ap_neg = []
post4ap_pos = []
post4ap_neg = []

# positive responders
print('\n\n------------------------------------------------')
print('PLOTTING: Avg. periphotostim positive responders')
print('------------------------------------------------')

ncols = 3
nrows = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
# fig2, axs2 = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            print(f"\nexperiment comparison: {allopticalResults.num_sig_responders_df.index[row]}")
            mean_pre4ap_ = allopticalResults.possig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.possig_responders_traces[row][1]
            print(f"# of mean_pre4ap traces: {len(mean_pre4ap_)}, and mean_post4ap traces: {len(mean_post4ap_)}")

            if len(mean_pre4ap_) > 1 and len(mean_post4ap_) > 1:
                ax = axs[counter//ncols, counter % ncols]

                meanst = np.mean(mean_pre4ap_, axis=0)
                ## change xaxis to time (secs)
                if len(meanst) < 100:
                    fps = 15
                else:
                    fps = 30

                fig, ax = aoplot.plot_periphotostim_avg2(dataset=[mean_pre4ap_, mean_post4ap_], fps=fps, legend_labels=[f"pre4ap {len(mean_pre4ap_)} cells", f"post4ap {len(mean_post4ap_)} cells"],
                                               colors=['black', 'green'], avg_with_std=True, title=f"{allopticalResults.num_sig_responders_df.index[row]}", ylim=[-0.5, 1.0],
                                               pre_stim_sec=allopticalResults.pre_stim_sec, fig=fig, ax=ax, show=False, fontsize='small',
                                                         xlabel=None, ylabel=None)

            counter += 1
axs[0, 0].set_ylabel('dF/stdF')
axs[0, 0].set_xlabel('Time post stim (secs)')
title = 'Avg. periphotostim positive responders'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()

# fig2.suptitle('Summed response of positive responders')
# fig2.show()


# negative responders
print('\n\n------------------------------------------------')
print('PLOTTING: Avg. periphotostim negative responders')
print('------------------------------------------------')
ncols = 3
nrows = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            print(f"\nexperiment comparison: {allopticalResults.num_sig_responders_df.index[row]}")
            mean_pre4ap_ = allopticalResults.negsig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.negsig_responders_traces[row][1]
            print(f"# of mean_pre4ap traces: {len(mean_pre4ap_)}, and mean_post4ap traces: {len(mean_post4ap_)}")

            if len(mean_pre4ap_) > 1 and len(mean_post4ap_) > 1:
                ax = axs[counter//ncols, counter % ncols]

                meanst = np.mean(mean_pre4ap_, axis=0)
                ## change xaxis to time (secs)
                if len(meanst) < 100:
                    fps = 15
                else:
                    fps = 30

                fig, ax = aoplot.plot_periphotostim_avg2(dataset=[mean_pre4ap_, mean_post4ap_], fps=fps, legend_labels=[f"pre4ap {len(mean_pre4ap_)} cells", f"post4ap {len(mean_post4ap_)} cells"],
                                               colors=['black', 'red'], avg_with_std=True, title=f"{allopticalResults.num_sig_responders_df.index[row]}", ylim=[-0.5, 1.0],
                                               pre_stim_sec=allopticalResults.pre_stim_sec, fig=fig, ax=ax, show=False, fontsize='small',
                                                         xlabel=None, ylabel=None)

            counter += 1
axs[0, 0].set_ylabel('dF/stdF')
axs[0, 0].set_xlabel('Time post stim (secs)')
title = 'Avg. periphotostim negative responders'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()


# %% 5.3.3) PLOT - summed photostim response - NON TARGETS
experiments = ['RL108t', 'RL109t', 'PS05t', 'PS07t', 'PS06t', 'PS11t']
pre4ap_pos = []
pre4ap_neg = []
post4ap_pos = []
post4ap_neg = []

## dataframe for saving measurement of AUC of total evoked responses
auc_responders = pd.DataFrame(columns=['pre4ap_pos_auc', 'pre4ap_neg_auc', 'post4ap_pos_auc', 'post4ap_neg_auc'],
                              index=allopticalResults.num_sig_responders_df.index)

# positive responders
print('\n\n------------------------------------------------')
print('PLOTTING: Avg. total evoked activity positive responders')
print('------------------------------------------------')
ncols = 3
nrows = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            print(f"\nexperiment comparison: {allopticalResults.num_sig_responders_df.index[row]}")
            mean_pre4ap_ = allopticalResults.possig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.possig_responders_traces[row][1]
            normalize = [int(allopticalResults.num_sig_responders_df.iloc[row, -1])] * 2
            print(f"# of mean_pre4ap traces: {len(mean_pre4ap_)}, and mean_post4ap traces: {len(mean_post4ap_)}")

            if len(mean_pre4ap_) > 1 and len(mean_post4ap_) > 1:

                # plot avg with confidence intervals
                # fig, ax = plt.subplots()

                ax = axs[counter//ncols, counter % ncols]

                meanst = np.mean(mean_pre4ap_, axis=0)
                ## change xaxis to time (secs)
                if len(meanst) < 100:
                    fps = 15
                else:
                    fps = 30

                fig, ax, auc = aoplot.plot_periphotostim_addition(dataset=[mean_pre4ap_, mean_post4ap_], normalize=normalize, fps=fps,
                                                                  legend_labels=[f"pre {mean_pre4ap_.shape[0]} cells", f"post {mean_post4ap_.shape[0]} cells"],
                                                                  colors=['black', 'green'], xlabel=None, ylabel=None,
                                                                  avg_with_std=True,  title=f"{allopticalResults.num_sig_responders_df.index[row]}",
                                                                  ylim=None, pre_stim_sec=allopticalResults.pre_stim_sec, fig=fig, ax=ax, show=False,
                                                                  fontsize='x-small')

                auc_responders.loc[allopticalResults.num_sig_responders_df.index[row], ['pre4ap_pos_auc', 'post4ap_pos_auc']] = auc

            counter += 1
axs[0, 0].set_ylabel('norm. total response (a.u.)')
axs[0, 0].set_xlabel('Time post stim (secs)')
title = 'Summed response of positive responders'
fig.tight_layout()
fig.suptitle(title)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()


# negative responders
print('\n\n------------------------------------------------')
print('PLOTTING: Avg. total evoked activity negative responders')
print('------------------------------------------------')
ncols = 3
nrows = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
# fig2, axs2 = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            print(f"\nexperiment comparison: {allopticalResults.num_sig_responders_df.index[row]}")
            mean_pre4ap_ = allopticalResults.negsig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.negsig_responders_traces[row][1]

            # plot avg with confidence intervals
            # fig, ax = plt.subplots()
            print(f"# of mean_pre4ap traces: {len(mean_pre4ap_)}, and mean_post4ap traces: {len(mean_post4ap_)}")

            if len(mean_pre4ap_) > 1 and len(mean_post4ap_) > 1:

                ax = axs[counter // ncols, counter % ncols]

                meanst = np.mean(mean_pre4ap_, axis=0)
                ## change xaxis to time (secs)
                if len(meanst) < 100:
                    fps = 15
                else:
                    fps = 30

                fig, ax, auc = aoplot.plot_periphotostim_addition(dataset=[mean_pre4ap_, mean_post4ap_],
                                                                  normalize=normalize, fps=fps,
                                                                  legend_labels=[f"pre {mean_pre4ap_.shape[0]} cells", f"post {mean_post4ap_.shape[0]} cells"],
                                                                  colors=['black', 'red'], xlabel=None, ylabel=None, avg_with_std=True,
                                                                  title=f"{allopticalResults.num_sig_responders_df.index[row]}",
                                                                  ylim=None, pre_stim_sec=allopticalResults.pre_stim_sec,
                                                                  fig=fig, ax=ax, show=False, fontsize='x-small')

                auc_responders.loc[
                    allopticalResults.num_sig_responders_df.index[row], ['pre4ap_neg_auc', 'post4ap_neg_auc']] = auc

            counter += 1
axs[0, 0].set_ylabel('norm. total response (a.u.)')
axs[0, 0].set_xlabel('Time post stim (secs)')
title = 'Summed response of negative responders'
fig.suptitle(title)
fig.tight_layout()
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()

allopticalResults.auc_responders_df = auc_responders

allopticalResults.save()


# %% 5.3.3.1) PLOT - # BARPLOT OF AUC OF TOTAL EVOKED PHOTOSTIM AVG ACITIVTY

print('\n\n------------------------------------------------')
print('PLOTTING: AUC OF TOTAL EVOKED PHOTOSTIM AVG ACTIVITY')
print('------------------------------------------------')

data=[]
cols = ['pre4ap_pos_auc', 'post4ap_pos_auc']
for col in cols:
    data.append(list(allopticalResults.auc_responders_df.loc[:, col]))

cols = ['pre4ap_neg_auc', 'post4ap_neg_auc']
for col in cols:
    data.append(list(allopticalResults.auc_responders_df.loc[:, col]))

print(allopticalResults.auc_responders_df)

experiments = ['RL108', 'RL109', 'PS05', 'PS07', 'PS06', 'PS11']
pre4ap_pos_auc = []
pre4ap_neg_auc = []
post4ap_pos_auc = []
post4ap_neg_auc = []

for exp in experiments:
    rows = []
    for row in range(len(allopticalResults.auc_responders_df.index)):
        if exp in allopticalResults.auc_responders_df.index[row]:
            rows.append(row)
    x = allopticalResults.auc_responders_df.iloc[rows, :].mean(axis=0)
    pre4ap_pos_auc.append(x[0])
    pre4ap_neg_auc.append(x[1])
    post4ap_pos_auc.append(x[2])
    post4ap_neg_auc.append(x[3])


fig, axs = plt.subplots(ncols=2, nrows=1, figsize=[4,3])
data = [pre4ap_pos_auc, post4ap_pos_auc]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap', 'post4ap'], colors=['lightgreen', 'forestgreen'],
                        bar=False, paired=True, expand_size_x=0.4, expand_size_y=1.2, title='pos responders',
                        y_label='norm. evoked activity (a.u.)', fig=fig, ax=axs[0], show=False, shrink_text=0.7)

data = [pre4ap_neg_auc, post4ap_neg_auc]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap', 'post4ap'], colors=['skyblue', 'steelblue'],
                        bar=False, paired=True, expand_size_x=0.5, expand_size_y=1.2, title='neg responders',
                        y_label='norm. evoked activity (a.u.)', fig=fig, ax=axs[1], show=False, shrink_text=0.7)
title = 'network evoked photostim activity - nontargets - pre vs. post4ap'
fig.suptitle(title, fontsize=8.5)
save_path = save_path_prefix + f"/{title}.png"
plt.savefig(save_path)
fig.show()



"""# 5.5.3) # # -  total post stim response evoked across all cells recorded
    # - like maybe add up all trials (sig and non sig), and all cells
    # - and compare pre-4ap and post-4ap (exp by exp, maybe normalizing the peak value per comparison from pre4ap?)
    # - or just make one graph per comparison and show all to Adam?
"""


# %% 5.4-todo) PLOT - plot some response measure against success rate of the stimulation
"""#  think about some normalization via success rate of the stimulus (plot some response measure against success rate of the stimulation) - 
#  calculate pearson's correlation value of the association
"""
# %% 5.5-todo) PLOT - dynamic changes in responses across multiple stim trials - this is very similar to the deltaActivity measurements

"""
dynamic changes in responses across multiple stim trials - this is very similar to the deltaActivity measurements
- NOT REALLY APPROPRIATE HERE, THIS IS A WHOLE NEW SET OF ANALYSIS
"""

# %% 5.6-dc) PLOTting- responses of non targets to photostim - xyloc 2D plot using s2p ROI colors

# xyloc plot of pos., neg. and non responders -- NOT SURE IF ITS WORKING PROPERLY RIGHT NOW, NOT WORTH THE EFFORT RIGHT NOW LIKE THIS. NOT THE FULL WAY TO MEASURE SPATIAL RELATIONSHIPS AT ALL AS WELL.
expobj.dfstdf_nontargets = pd.DataFrame(expobj.post_array_responses, index=expobj.s2p_nontargets, columns=expobj.stim_start_frames)
df = pd.DataFrame(expobj.post_array_responses[expobj.sig_units, :], index=[expobj.s2p_nontargets[i] for i, x in enumerate(expobj.sig_units) if x], columns=expobj.stim_start_frames)
s_ = np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)
df = pd.DataFrame(expobj.post_array_responses[s_, :][0], index=[expobj.s2p_nontargets[i] for i in s_[0]], columns=expobj.stim_start_frames)
aoplot.xyloc_responses(expobj, df=df, clim=[-1, +1], plot_target_coords=True)










# %% 1) plot responses of SLM TARGETS in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
# #  - with option to plot only successful or only failure stims!

### 1.1) PRE-4AP TRIALS
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





# %% ## 1.2) POST-4AP TRIALS - IN SZ STIMS - EXCLUDE STIMS/CELLS INSIDE SZ BOUNDARY
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



# %% ## 1.3) POST-4AP TRIALS (OUT SZ STIMS)

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






# %% 1.3.1)
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


# %% 1.4) COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

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
                    expobj.StimSuccessRate_SLMtargets_outsz, expobj.hits_SLMtargets_outsz, expobj.responses_SLMtargets_outsz, expobj.traces_SLMtargets_successes_outsz = \
                        expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_out_sz)
                    success_responses = expobj.hits_SLMtargets_outsz * expobj.responses_SLMtargets_outsz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response outsz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())

                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_in_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (insz) [1.4.3] ***********')
                    expobj.StimSuccessRate_SLMtargets_insz, expobj.hits_SLMtargets_insz, expobj.responses_SLMtargets_insz, expobj.traces_SLMtargets_successes_insz = \
                        expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_in_sz)

                    success_responses = expobj.hits_SLMtargets_insz * expobj.responses_SLMtargets_insz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response insz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())


            elif 'pre' in expobj.metainfo['exptype']:
                seizure_filter = False
                print('\n Calculating stim success rates and response magnitudes [1.4.4] ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
                    expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stim_start_frames)

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


# %% 1.5) COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

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








# %% 2) BAR PLOT FOR PHOTOSTIM RESPONSE MAGNITUDE B/W PRE AND POST 4AP TRIALS
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

# %% 4) plot peri-photostim avg traces for all trials analyzed to make sure they look alright
# -- plot as little postage stamps

# plot avg of successes in green
# plot avg of failures in gray
# plot line at dF_stdF = 0.3
# add text in plot for avg dF_stdF value of successes, and % of successes



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

# for i in range(len(response_traces_successes[cell][:-1])):
#     plt.plot(response_traces_successes[cell][:-1][i])
# plt.show()

# for i in responses_magnitudes_successes.keys():
#     print(len(responses_magnitudes_successes))

# plot barplot with points only comparing response magnitude of successes

# %% 6) 4D plotting of the seizure wavefront (2D space vs. time vs. Flu intensity)

# import expobj
expobj, experiment = aoutils.import_expobj(aoresults_map_id='post h.0', do_processing=False)
aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Frames', figsize=[20, 3])


downsampled_tiff_path = f"{expobj.tiff_path[:-4]}_downsampled.tif"

# get the 2D x time Flu array
print(f"loading tiff from: {downsampled_tiff_path}")
im_stack = tf.imread(downsampled_tiff_path)
print('|- Loaded experiment tiff of shape: ', im_stack.shape)

# binning down the imstack to a reasonable size
im_binned_ = resize(im_stack, (im_stack.shape[0], 100, 100))

# building up the flat lists to use for 3D plotting....

start = 10500//4
end = 16000//4
sub = end - start

im_binned = im_binned_[start:end, :, :]

x_size=im_binned.shape[1]
y_size=im_binned.shape[2]
t_size=sub

time = np.asarray([np.array([[i]*x_size]*y_size) for i in range(t_size)])

x = np.asarray(list(range(x_size))*y_size*t_size)
y = np.asarray([i_y for i_y in range(y_size) for i_x in range(x_size)] * t_size)
t = time.flatten()
flu = im_binned.flatten()

im_array = np.array([x, y, t], dtype=np.float)

assert len(x) == len(y) == len(t), print(f'length mismatch between x{len(x)}, y{len(y)}, t{len(t)}')

# plot 3D projection scatter plot
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
ax.scatter(im_array[2], im_array[1], im_array[0], c=flu, cmap='Reds', linewidth=0.5, alpha=0.005)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.grid(False)

ax.set_xlabel('time (frames)')
ax.set_ylabel('y axis')
ax.set_zlabel('x axis')
fig.tight_layout()
fig.show()


# %% 8.0-main-dc) TODO collect targets responses for stims vs. distance (starting with old code)- low priority right now

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






# %% 9.0-main) avg responses around photostim targets - pre vs. post4ap

