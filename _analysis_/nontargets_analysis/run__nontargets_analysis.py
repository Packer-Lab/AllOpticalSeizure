import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt


from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
import funcsforprajay.plotting as pplot
import funcsforprajay.funcs as pj

import pandas as pd
import numpy as np

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults
from _main_.AllOpticalMain import alloptical

import _alloptical_utils as Utils

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

############################## run processing/analysis/plotting: #######################################################
# %% 5) RESPONSES of nontargets VS. DISTANCE TO NEAREST TARGET AND DISTANCE TO SZ WAVEFRONT
"""
Objectives:
[ ] - scatter plot of nontargets responses (z scored (to baseline)) vs., distance to nearest targets and distance to sz wavefront
[ ] - binned 2D histogram of nontargets responses (z scored (to baseline)) vs., distance to nearest targets and distance to sz wavefront

"""

# 5.0) run processing + create dataframe of nontargets responses across stim groups and distances to targets:
results.collect_nontargets_stim_responses(run_post4ap=True)

# 5.2) binning responses relative to distance from targets, then average the responses across binned distances

# run as a results method function
# results.binned_distances_vs_responses(measurement='new influence response')
results.binned_distances_vs_responses_interictal(measurement='new influence response')

# %% 5.0) plotting
# plot hist distribution of distances to target
baseline_responses = results.baseline_responses.iloc[results.pre4ap_idxs]


# num occurrences at each distance - split by trial types
fig, ax = plt.subplots(figsize = (5,5))
distances = []
for exp in np.unique(baseline_responses['expID']):
    _distances = list(baseline_responses[baseline_responses['expID'] == exp]['distance target'])
    distances.append(_distances)
ax.hist(distances, 40, density=True, histtype='bar', stacked=True)
ax.set_title('density of measurements by individual experiments')
ax.set_xlabel('distance to target (um)')
fig.show()




# %% 5.2.2) make plot of average responses +/- std across space bins - interictal

measurement = 'new influence response'

fig, ax = plt.subplots(figsize = (4, 4))

ax.axhline(y=0, ls='--', color='black', lw=1)
ax.axvline(x=20, ls='--', color='black', lw=1)

# binned shuffled distances vs. responses
distances = results.binned_distance_vs_responses_shuffled_interictal[measurement]['distances']
avg_binned_responses = results.binned_distance_vs_responses_shuffled_interictal[measurement]['avg binned responses']
std_binned_responses = results.binned_distance_vs_responses_shuffled_interictal[measurement]['std binned responses']
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + std_binned_responses), y2=list(avg_binned_responses - std_binned_responses), alpha=0.1, color='orange')
ax.plot(distances, avg_binned_responses, lw=3, color='orange')


# binned distances vs responses
distances = results.binned_distance_vs_responses_interictal[measurement]['distances']
avg_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['avg binned responses']
std_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['std binned responses']
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + std_binned_responses), y2=list(avg_binned_responses - std_binned_responses), alpha=0.1, color='royalblue')
ax.plot(distances, avg_binned_responses, lw=3, color='royalblue')



ax.set_title(f"{measurement} vs. distance to target - interictal", wrap=True)
ax.set_xlim([0, 600])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='distance to target (um)', y_label='influence of photostim')
fig.tight_layout()

fig.show()


# %% 5.2.1) make plot of average responses +/- std across space bins - baseline

measurement = 'new influence response'

fig, ax = plt.subplots(figsize = (4, 4))

ax.axhline(y=0, ls='--', color='black', lw=1)
ax.axvline(x=20, ls='--', color='black', lw=1)

# binned shuffled distances vs. responses
distances = results.binned_distance_vs_responses_shuffled[measurement]['distances']
avg_binned_responses = results.binned_distance_vs_responses_shuffled[measurement]['avg binned responses']
std_binned_responses = results.binned_distance_vs_responses_shuffled[measurement]['std binned responses']
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + std_binned_responses), y2=list(avg_binned_responses - std_binned_responses), alpha=0.1, color='orange')
ax.plot(distances, avg_binned_responses, lw=3, color='orange')


# binned distances vs responses
distances = results.binned_distance_vs_responses[measurement]['distances']
avg_binned_responses = results.binned_distance_vs_responses[measurement]['avg binned responses']
std_binned_responses = results.binned_distance_vs_responses[measurement]['std binned responses']
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + std_binned_responses), y2=list(avg_binned_responses - std_binned_responses), alpha=0.1, color='royalblue')
ax.plot(distances, avg_binned_responses, lw=3, color='royalblue')



ax.set_title(f"{measurement} vs. distance to target (um)", wrap=True)
ax.set_xlim([0, 600])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='distance to target (um)', y_label='influence of photostim')
fig.tight_layout()

fig.show()







# %% 5.1) plotting scatter plots of responses - BASELINE

# fig, ax = plt.subplots(figsize=(4, 4))
# sns.scatterplot(data=results.responses, x="distance target", y="distance sz", hue='z score response', ax=ax, hue_norm=(-6, 6),
#                 legend='brief')
# fig.show()





fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(results.baseline_responses["distance target"][results.post4ap_idxs], results.baseline_responses["distance sz"][results.post4ap_idxs], color='gray', alpha=0.01,
           s=5)
ax.set_title('distance to target vs. distance to sz - ictal')
fig.tight_layout(pad=1.3)
fig.show()


fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(results.baseline_responses["distance target"][results.pre4ap_idxs], results.baseline_responses["z score response"][results.pre4ap_idxs], color='red', alpha=0.01,
           s=5)
ax.set_title('distance to target vs. z score response - baseline')
fig.tight_layout(pad=1.3)
fig.show()


fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(results.baseline_responses["distance target"][results.post4ap_idxs], results.baseline_responses["z score response"][results.post4ap_idxs], color='red', alpha=0.01,
           s=5)
ax.set_title('distance to target vs. z score response - ictal')
fig.tight_layout(pad=1.3)
fig.show()


fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(results.baseline_responses["distance sz"][results.post4ap_idxs], results.baseline_responses["z score response"][results.post4ap_idxs], color='orange', alpha=0.01,
           s=5)
fig.tight_layout(pad=1.3)
ax.set_title('distance sz vs. distance to target - ictal')
fig.show()



# %% 5.1.1) trying paired plot with seaborn -- takes too long to run actually

sns.pairplot(data=results.baseline_responses, vars=['distance sz', 'distance target', 'z score response'])
fig.show()




# %% 0) processing alloptical photostim and fakestim responses

# main.run__initPhotostimResponsesAnalysisNonTargets()
# main.run__fakestims_processing()

# main.run__plot_sig_responders_traces(plot_baseline_responders=True)
# main.run__plot_sig_responders_traces(plot_baseline_responders=False)
# main.run__create_anndata()
#
# # 2) basic plotting of responders baseline, interictal, and ictal
# main.collect__avg_magnitude_response(results=results, collect_baseline_responders=True)
# main.plot__avg_magnitude_response(results=results, plot_baseline_responders=True)
#
# main.collect__avg_magnitude_response(results=results, collect_baseline_responders=False)
# main.plot__avg_magnitude_response(results=results,
#                                   plot_baseline_responders=False)  # < - there is one experiment that doesn't have any responders i think....
#
# main.collect__avg_num_response(results=results)
# main.plot__avg_num_responders(results=results)


# %% 2) plotting alloptical and fakestim responses
# PhotostimResponsesAnalysisNonTargets.run__plot_sig_responders_traces(plot_baseline_responders=False)

# %% 3) collecting all summed nontargets photostim and fakestim responses vs. total targets photostim and fakestim responses
main.run__summed_responses(rerun=1)


# %% 3.1) plotting exps total nontargets photostim (and fakestim) responses vs. total targets photostim (and fakestim) responses

# PhotostimResponsesAnalysisNonTargets.plot__exps_summed_nontargets_vs_summed_targets()

# %% 3.2) collecting and plotting z scored summed total and nontargets responses

main.collect__zscored_summed_activity_vs_targets_activity(results=results)
# main.plot__summed_activity_vs_targets_activity(results=results)


# %% 4) ANALYSIS OF TOTAL EVOKED RESPONSES OF NONTARGETS DURING ICTAL PHASE #################################################################
"""
objectives:
- scatter plot: total responses of targets in proximal vs. mean z scored (to baseline) responses in proximal
- scatter plot: total responses of targets in distal vs. mean z scored (to baseline) responses of nontargets in distal
"""

import sys

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
    PhotostimResponsesNonTargetsResults, \
    PhotostimResponsesQuantificationNonTargets

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot


# 4.0) collect nontargets activity - split up proximal vs. distal
def calculate__mean_responses_nontargets(expobj: Post4ap):
    """calculate mean z scored (to baseline) responses in proximal, and in distal, of nontargets."""

    assert 'post' in expobj.exptype, 'pre4ap trials not allowed.'

    nontargets_responses = expobj.PhotostimResponsesNonTargets
    assert 'nontargets responses z scored (to baseline)' in nontargets_responses.adata.layers.keys(), 'nontargets responses z scored (to baseline) not found as layer in nontargets_responses adata table.'

    # proximal nontargets
    nontargets_mean_zscore = [np.nan for i in range(nontargets_responses.adata.n_vars)]
    for stim_idx in np.where(nontargets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][
            expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location'][:, stim_idx] == 'proximal']
        if len(cells_) > 0:
            proximal_idx = [idx for idx, cell in enumerate(nontargets_responses.adata.obs['original_index']) if
                            cell in list(cells_)]
            nontargets_mean_zscore[stim_idx] = np.nanmean(
                nontargets_responses.adata.layers['nontargets responses z scored (to baseline)'][
                    proximal_idx, stim_idx], axis=0)

    nontargets_responses.adata.add_variable(var_name='nontargets - proximal - mean z score',
                                            values=nontargets_mean_zscore)

    # distal nontargets
    nontargets_mean_zscore = [np.nan for i in range(nontargets_responses.adata.n_vars)]
    for stim_idx in np.where(nontargets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][
            expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location'][:, stim_idx] == 'distal']
        if len(cells_) > 0:
            distal_idx = [idx for idx, cell in enumerate(nontargets_responses.adata.obs['original_index']) if
                          cell in list(cells_)]
            nontargets_mean_zscore[stim_idx] = np.nanmean(
                nontargets_responses.adata.layers['nontargets responses z scored (to baseline)'][distal_idx, stim_idx],
                axis=0)

    nontargets_responses.adata.add_variable(var_name='nontargets - distal - mean z score',
                                            values=nontargets_mean_zscore)

    return nontargets_responses


# 4.0.1) collect targets activity - split up proximal vs. distal mean right now
def calculate__summed_responses_targets(expobj: Post4ap):
    """calculate total summed dFF responses of SLM targets of experiments to compare with summed responses of nontargets."""

    assert 'post' in expobj.exptype, 'pre4ap trials not allowed.'

    targets_responses = expobj.PhotostimResponsesSLMTargets

    # proximal targets
    targets_summed_zscore = [np.nan for i in range(targets_responses.adata.n_vars)]
    for stim_idx in np.where(targets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index[
            expobj.TargetsSzInvasionSpatial_codereview.adata.layers['outsz location'][:, stim_idx] == 'proximal']
        if len(cells_) > 0:
            proximal_idx = [int(idx) for idx, cell in
                            enumerate(expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index) if
                            cell in list(cells_)]
            targets_summed_zscore[stim_idx] = np.nanmean(targets_responses.adata.X[proximal_idx, stim_idx], axis=0)

        # troubleshooting: where is the sum targets = 0 for certain stims coming from?
        if np.round(targets_summed_zscore[stim_idx], 1) == 0.0:
            print('break here')

    targets_responses.adata.add_variable(var_name='targets - proximal - total z score', values=targets_summed_zscore)

    # distal targets
    targets_summed_zscore = [np.nan for i in range(targets_responses.adata.n_vars)]
    for stim_idx in np.where(targets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index[
            expobj.TargetsSzInvasionSpatial_codereview.adata.layers['outsz location'][:, stim_idx] == 'distal']
        if len(cells_) > 0:
            distal_idx = [int(idx) for idx, cell in
                          enumerate(expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index) if cell in list(cells_)]
            targets_summed_zscore[stim_idx] = np.nanmean(targets_responses.adata.X[distal_idx, stim_idx], axis=0)

    targets_responses.adata.add_variable(var_name='targets - distal - total z score', values=targets_summed_zscore)

    return targets_responses


# 4.1) collect results of targets and non targets activity across all exps - split up proximal vs. distal
def collect__mean_nontargets_activity_vs_targets_activity(results: PhotostimResponsesNonTargetsResults, rerun=1):
    """collect mean zscored activity of nontargets proximal and distal to sz wavefront, and also total targets activity"""

    # post4ap - interictal #############################################################################################
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=rerun,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
    def collect_total_responses_ictal(**kwargs):
        """collect z scored (to baseline) summed responses for photostim nontargets, split by proximal and distal groups"""
        expobj: Post4ap = kwargs['expobj']

        summed_responses_proximal = pd.DataFrame({'exp': [expobj.t_series_name] * sum(
            [expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'][0]),
                                                  'targets': expobj.PhotostimResponsesSLMTargets.adata.var[
                                                      'targets - proximal - total z score'][
                                                      expobj.PhotostimResponsesSLMTargets.adata.var[
                                                          'stim_group'] == 'ictal'],
                                                  'non-targets': expobj.PhotostimResponsesNonTargets.adata.var[
                                                      'nontargets - proximal - mean z score'][
                                                      expobj.PhotostimResponsesSLMTargets.adata.var[
                                                          'stim_group'] == 'ictal'],
                                                  })

        summed_responses_distal = pd.DataFrame({'exp': [expobj.t_series_name] * sum(
            [expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'][0]),
                                                'targets': expobj.PhotostimResponsesSLMTargets.adata.var[
                                                    'targets - distal - total z score'][
                                                    expobj.PhotostimResponsesSLMTargets.adata.var[
                                                        'stim_group'] == 'ictal'],
                                                'non-targets': expobj.PhotostimResponsesNonTargets.adata.var[
                                                    'nontargets - distal - mean z score'][
                                                    expobj.PhotostimResponsesSLMTargets.adata.var[
                                                        'stim_group'] == 'ictal']
                                                })

        # calculating linear regression metrics between summed targets and summed total network for each experiment
        # proximal cells
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=summed_responses_proximal['targets'],
                                                                       y=summed_responses_proximal['non-targets'])
        regression_y = slope * summed_responses_proximal['targets'] + intercept

        lin_reg_scores_proximal = pd.DataFrame({
            'exp': expobj.t_series_name,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value
        }, index=[expobj.t_series_name])

        # distal cells
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=summed_responses_distal['targets'],
                                                                       y=summed_responses_distal['non-targets'])
        regression_y = slope * summed_responses_distal['targets'] + intercept

        lin_reg_scores_distal = pd.DataFrame({
            'exp': expobj.t_series_name,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value
        }, index=[expobj.t_series_name])

        return summed_responses_proximal, summed_responses_distal, lin_reg_scores_proximal, lin_reg_scores_distal

    func_collector_ictal = collect_total_responses_ictal()

    if func_collector_ictal is not None:

        summed_responses_proximal = pd.DataFrame({'exp': [], 'targets': [], 'non-targets': []})

        summed_responses_distal = pd.DataFrame({'exp': [], 'targets': [], 'non-targets': []})

        lin_reg_scores_proximal = pd.DataFrame(
            {'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})
        lin_reg_scores_distal = pd.DataFrame(
            {'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})

        for res in func_collector_ictal:
            summed_responses_proximal = pd.concat([summed_responses_proximal, res[0]])
            summed_responses_distal = pd.concat([summed_responses_distal, res[1]])
            lin_reg_scores_proximal = pd.concat([lin_reg_scores_proximal, res[2]])
            lin_reg_scores_distal = pd.concat([lin_reg_scores_distal, res[3]])

        print('summed responses proximal shape', summed_responses_proximal.shape)
        print('summed responses distal shape', summed_responses_distal.shape)

        results.summed_responses['ictal - proximal'] = summed_responses_proximal
        results.summed_responses['ictal - distal'] = summed_responses_distal
        results.lin_reg_summed_responses['ictal - proximal'] = lin_reg_scores_proximal
        results.lin_reg_summed_responses['ictal - distal'] = lin_reg_scores_distal
        results.save_results()


# 4.1.1) plot results of targets and non targets activity across all exps - split up proximal vs. distal
def plot__ictal_nontargets_vs_targets_activity(results: PhotostimResponsesNonTargetsResults):
    """scatter plot of stim trials comparing summed activity of targets (originally zscored to baseline) and mean activity of non-targets (original zscored to baseline).
    - split by proximal and distal locations.
    """
    # make plots

    # SCATTER PLOT OF DATAPOINTS
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    # PROXIMAL CELLS

    non_nan_idxs = []
    for idx in results.summed_responses['ictal - proximal'].index:
        l_ = list(results.summed_responses['ictal - proximal'].iloc[int(idx), :])
        nans = [np.isnan(x) for x in l_ if type(x) != str]
        if True not in nans:
            non_nan_idxs.append(int(idx))
        else:
            # print('debug here...')
            pass

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=results.summed_responses['ictal - proximal']['targets'][non_nan_idxs],
        y=results.summed_responses['ictal - proximal']['non-targets'][non_nan_idxs])
    regression_y = slope * results.summed_responses['ictal - proximal']['targets'][non_nan_idxs] + intercept
    fig, axs[0] = pplot.make_general_scatter(
        x_list=[results.summed_responses['ictal - proximal']['targets'][non_nan_idxs]],
        y_data=[results.summed_responses['ictal - proximal']['non-targets'][non_nan_idxs]], fig=fig, ax=axs[0],
        s=50, facecolors=['white'], edgecolors=['blue'], lw=1, alpha=0.5,
        x_labels=['mean targets activity (z scored)'], y_labels=['mean non-targets activity (z scored)'],
        legend_labels=[f'proximal cells - $R^2$: {r_value ** 2:.2e}, p = {p_value ** 2:.2e}, $m$ = {slope:.2e}'],
        show=False)
    axs[0].plot(results.summed_responses['ictal - proximal']['targets'][non_nan_idxs], regression_y, color='royalblue',
                lw=2)

    # DISTAL CELLS

    non_nan_idxs = []
    for idx in results.summed_responses['ictal - distal'].index:
        l_ = list(results.summed_responses['ictal - distal'].iloc[int(idx), :])
        nans = [np.isnan(x) for x in l_ if type(x) != str]
        if True not in nans:
            non_nan_idxs.append(int(idx))
        else:
            # print('debug here...')
            pass

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=results.summed_responses['ictal - distal']['targets'][non_nan_idxs],
        y=results.summed_responses['ictal - distal']['non-targets'][non_nan_idxs])

    regression_y = slope * results.summed_responses['ictal - distal']['targets'][non_nan_idxs] + intercept

    pplot.make_general_scatter(x_list=[results.summed_responses['ictal - distal']['targets'][non_nan_idxs]],
                               y_data=[results.summed_responses['ictal - distal']['non-targets'][non_nan_idxs]], s=50,
                               facecolors=['white'], edgecolors=['green'], lw=1, alpha=0.5,
                               x_labels=['mean targets activity (zscored)'],
                               y_labels=['mean non-targets activity (z scored)'], fig=fig, ax=axs[1],
                               legend_labels=[
                                   f'distal cells - $R^2$: {r_value ** 2:.2e}, p = {p_value ** 2:.2e}, $m$ = {slope:.2e}'],
                               show=False)

    axs[1].plot(results.summed_responses['ictal - distal']['targets'][non_nan_idxs], regression_y, color='forestgreen',
                lw=2)

    # PLOTTING OPTIONS
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[0].set_xlim([-100, 100])
    axs[1].set_xlim([-100, 100])
    fig.suptitle('Total z-scored (to baseline) responses for across all exps', wrap=True)
    fig.tight_layout(pad=0.6)
    fig.show()


# %%

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0,
                                skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS, )
# run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
def run__ictal_nontargets_responses_processing(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.PhotostimResponsesNonTargets = calculate__mean_responses_nontargets(expobj=kwargs['expobj'])
    expobj.PhotostimResponsesSLMTargets = calculate__summed_responses_targets(expobj=kwargs['expobj'])
    expobj.save()


run__ictal_nontargets_responses_processing()

collect__mean_nontargets_activity_vs_targets_activity(results=results, rerun=1)

plot__ictal_nontargets_vs_targets_activity(results=results)

# %%
