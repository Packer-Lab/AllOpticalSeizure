# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS - FOCUS ON FOLLOWERS!
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

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-11'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre e.1')  # PLACEHOLDER IMPORT OF EXPOBJ TO MAKE THE CODE WORK


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


# %% 1.1.1-dc) PLOT - dF/F of significant pos. and neg. responders that were derived from dF/stdF method
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



# %% 1.1.2-dc) PLOT - creating large figures collating multiple plots describing responses of non targets to photostim for individual expobj's -- collecting code in aoutils.fig_non_targets_responses()
plot_subset = False

if plot_subset:
    selection = np.random.randint(0, expobj.dff_traces_avg.shape[0], 100)
else:
    selection = np.arange(expobj.dff_traces_avg.shape[0])

#### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
f = plt.figure(figsize=[30, 10])
gs = f.add_gridspec(2, 9)

# %% 1.1.2.1-dc) PLOT OF PERI-STIM AVG TRACES FOR ALL SIGNIFICANT AND NON-SIGNIFICANT RESPONDERS - also breaking down positive and negative responders

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

# %% 1.1.2.2-dc) PLOT - quantifying responses of non targets to photostim
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






# %% 1.2.1) PLOT - scatter plot 1) response magnitude vs. prestim std F, and 2) response magnitude vs. prestim mean F
## TODO check if these plots are coming out okay...

# 1.2.1.1) scatter plot response magnitude vs. prestim std F
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

# 1.2.1.2) scatter plot response magnitude vs. prestim mean F
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




# %% 1.2.2) PLOT - measuring avg raw pre-stim stdF for all non-targets - pre4ap vs. post4ap histogram comparison
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


# 1.2.2.1) PLOT - measuring avg raw pre-stim stdF for all non-targets - pre4ap only
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




# 1.2.2.2) PLOT - measuring avg raw pre-stim stdF for all non-targets - post4ap trials
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






# %% 1.2.3) PLOT - measuring avg. raw prestim F - do post4ap cells have a lower avg. raw prestim F?

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






# %% 1.3.1) PLOT - bar plot average # of significant responders (+ve and -ve) for pre vs. post 4ap
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


# %% 1.3.2) PLOT - peri-stim average response stim graph for positive and negative followers
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


# %% 1.3.3) PLOT - summed photostim response - NON TARGETS
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


# %% 1.3.3.1) PLOT - # BARPLOT OF AUC OF TOTAL EVOKED PHOTOSTIM AVG ACITIVTY

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



"""# 1.5.3) # # -  total post stim response evoked across all cells recorded
    # - like maybe add up all trials (sig and non sig), and all cells
    # - and compare pre-4ap and post-4ap (exp by exp, maybe normalizing the peak value per comparison from pre4ap?)
    # - or just make one graph per comparison and show all to Adam?
"""


# %% 1.4-todo) PLOT - plot some response measure against success rate of the stimulation
"""#  think about some normalization via success rate of the stimulus (plot some response measure against success rate of the stimulation) - 
#  calculate pearson's correlation value of the association
"""
# %% 1.5-todo) PLOT - dynamic changes in responses across multiple stim trials - this is very similar to the deltaActivity measurements

"""
dynamic changes in responses across multiple stim trials - this is very similar to the deltaActivity measurements
- NOT REALLY APPROPRIATE HERE, THIS IS A WHOLE NEW SET OF ANALYSIS
"""

# %% 1.6-dc) PLOTting- responses of non targets to photostim - xyloc 2D plot using s2p ROI colors

# xyloc plot of pos., neg. and non responders -- NOT SURE IF ITS WORKING PROPERLY RIGHT NOW, NOT WORTH THE EFFORT RIGHT NOW LIKE THIS. NOT THE FULL WAY TO MEASURE SPATIAL RELATIONSHIPS AT ALL AS WELL.
expobj.dfstdf_nontargets = pd.DataFrame(expobj.post_array_responses, index=expobj.s2p_nontargets, columns=expobj.stim_start_frames)
df = pd.DataFrame(expobj.post_array_responses[expobj.sig_units, :], index=[expobj.s2p_nontargets[i] for i, x in enumerate(expobj.sig_units) if x], columns=expobj.stim_start_frames)
s_ = np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)
df = pd.DataFrame(expobj.post_array_responses[s_, :][0], index=[expobj.s2p_nontargets[i] for i in s_[0]], columns=expobj.stim_start_frames)
aoplot.xyloc_responses(expobj, df=df, clim=[-1, +1], plot_target_coords=True)










