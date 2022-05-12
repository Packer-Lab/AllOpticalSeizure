# ARCHIVED!!!

"""


Current objectives:
- [ ]  scatter plot: total responses of targets in proximal vs. mean z scored (to baseline) responses in proximal
- [ ]  scatter plot: total responses of targets in distal vs. mean z scored (to baseline) responses of nontargets in distal


"""


import sys
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import PhotostimResponsesNonTargetsResults, \
    PhotostimResponsesQuantificationNonTargets

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# %%

# 4) ANALYSIS OF TOTAL EVOKED RESPONSES OF NONTARGETS DURING ICTAL PHASE #################################################################
# 4.0) collect nontargets activity - split up proximal vs. distal
def calculate__summed_responses_nontargets(expobj: Post4ap):
    """calculate mean z scored (to baseline) responses in proximal, and in distal, of nontargets."""

    assert 'post' in expobj.exptype, 'pre4ap trials not allowed.'

    nontargets_responses = expobj.PhotostimResponsesNonTargets
    assert 'nontargets responses z scored (to baseline)' in nontargets_responses.adata.layers.keys(), 'nontargets responses z scored (to baseline) not found as layer in nontargets_responses adata table.'

    # proximal nontargets
    nontargets_mean_zscore = [np.nan for i in range(nontargets_responses.adata.n_vars)]
    for stim_idx in np.where(nontargets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location'][:, stim_idx] == 'proximal']
        proximal_idx = [idx for idx, cell in enumerate(nontargets_responses.adata.obs['original_index']) if cell in cells_]
        nontargets_mean_zscore[stim_idx] = np.mean(nontargets_responses.adata.layers['nontargets responses z scored (to baseline)'][proximal_idx, stim_idx], axis=0)

    nontargets_responses.adata.add_variable(var_name='nontargets - proximal - mean z score', values=nontargets_mean_zscore)


    # distal nontargets
    nontargets_mean_zscore = [np.nan for i in range(nontargets_responses.adata.n_vars)]
    for stim_idx in np.where(nontargets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location'][:, stim_idx] == 'distal']
        distal_idx = [idx for idx, cell in enumerate(nontargets_responses.adata.obs['original_index']) if cell in cells_]
        nontargets_mean_zscore[stim_idx] = np.mean(nontargets_responses.adata.layers['nontargets responses z scored (to baseline)'][distal_idx, stim_idx], axis=0)

    nontargets_responses.adata.add_variable(var_name='nontargets - distal - mean z score',
                                            values=nontargets_mean_zscore)

    return nontargets_responses


def calculate__summed_responses_targets(expobj: Post4ap):
    """calculate total summed dFF responses of SLM targets of experiments to compare with summed responses of nontargets."""

    assert 'post' in expobj.exptype, 'pre4ap trials not allowed.'

    targets_responses = expobj.PhotostimResponsesSLMTargets

    # proximal targets
    targets_mean_zscore = [np.nan for i in range(targets_responses.adata.n_vars)]
    for stim_idx in np.where(targets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index[expobj.TargetsSzInvasionSpatial_codereview.adata.layers['outsz location'][:, stim_idx] == 'proximal']
        proximal_idx = [int(idx) for idx, cell in enumerate(expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index) if cell in cells_]
        targets_mean_zscore[stim_idx] = np.sum(targets_responses.adata.X[proximal_idx, stim_idx], axis=0)

    targets_responses.adata.add_variable(var_name='targets - proximal - total z score', values=targets_mean_zscore)


    # distal targets
    targets_mean_zscore = [np.nan for i in range(targets_responses.adata.n_vars)]
    for stim_idx in np.where(targets_responses.adata.var['stim_group'] == 'ictal')[0]:
        cells_ = expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index[expobj.TargetsSzInvasionSpatial_codereview.adata.layers['outsz location'][:, stim_idx] == 'distal']
        distal_idx = [int(idx) for idx, cell in enumerate(expobj.TargetsSzInvasionSpatial_codereview.adata.obs.index) if cell in cells_]
        targets_mean_zscore[stim_idx] = np.sum(targets_responses.adata.X[distal_idx, stim_idx], axis=0)

    targets_responses.adata.add_variable(var_name='targets - distal - total z score', values=targets_mean_zscore)

    return targets_responses



def __collect__summed_activity_vs_targets_activity(results: PhotostimResponsesNonTargetsResults, rerun = 0):
    """collect summed zscored activity of nontargets proximal and distal to sz wavefront"""

    # post4ap - interictal #############################################################################################
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=rerun,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
    def collect_summed_responses_ictal(**kwargs):
        """collect z scored (to baseline) summed responses for photostim nontargets, split by proximal and distal groups"""
        expobj: Post4ap = kwargs['expobj']

        summed_responses_proximal = pd.DataFrame({'exp': [expobj.t_series_name] * sum([expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'][0]),
                                         'targets': expobj.PhotostimResponsesSLMTargets.adata.var['targets - proximal - total z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'],
                                         'non-targets': expobj.PhotostimResponsesNonTargets.adata.var['nontargets - proximal - mean z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'],
                                         })

        summed_responses_distal = pd.DataFrame({'exp': [expobj.t_series_name] * sum([expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'][0]),
                                         'targets': expobj.PhotostimResponsesSLMTargets.adata.var['targets - distal - total z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'],
                                         'non-targets': expobj.PhotostimResponsesNonTargets.adata.var['nontargets - distal - mean z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal']
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

    func_collector_ictal = collect_summed_responses_ictal()

    if func_collector_ictal is not None:

        summed_responses_proximal = pd.DataFrame({'exp': [], 'targets': [], 'nontargets': []})

        summed_responses_distal = pd.DataFrame({'exp': [], 'targets': [], 'nontargets': []})

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


def __plot__ictal_nontargets_vs_targets_activity(results: PhotostimResponsesNonTargetsResults):
    """scatter plot of stim trials comparing summed activity of targets (originally zscored to baseline) and mean activity of nontargets (original zscored to baseline).
    - split by proximal and distal locations.
    """
    # make plots

    # SCATTER PLOT OF DATAPOINTS
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    # PROXIMAL CELLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=results.summed_responses['ictal - proximal']['targets'],
                                                                   y=results.summed_responses['ictal - proximal']['nontargets'])
    regression_y = slope * results.summed_responses['ictal - proximal']['targets'] + intercept
    fig, axs[0] = pplot.make_general_scatter(
        x_list=[results.summed_responses['ictal - proximal']['targets']],
        y_data=[results.summed_responses['ictal - proximal']['nontargets']], fig=fig, ax=axs[0],
        s=50, facecolors=['white'], edgecolors=['blue'], lw=1, alpha=0.5,
        x_labels=['total targets activity (z scored)'], y_labels=['mean nontargets activity (z scored)'],
        legend_labels=[f'proximal cells - $R^2$: {r_value ** 2:.2e}, p = {p_value ** 2:.2e}, $m$ = {slope:.2e}'],
        show=False)
    axs[0].plot(results.summed_responses['ictal - proximal']['targets'], regression_y, color='royalblue', lw=2)

    # DISTAL CELLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=results.summed_responses['ictal - distal']['targets'],
        y=results.summed_responses['ictal - distal']['nontargets'])

    regression_y = slope * results.summed_responses['ictal - distal']['targets'] + intercept

    pplot.make_general_scatter(x_list=[results.summed_responses['ictal - distal']['targets']],
                               y_data=[results.summed_responses['ictal - distal']['nontargets']], s=50,
                               facecolors=['white'], edgecolors=['green'], lw=1, alpha=0.5, x_labels=['total targets activity'],
                               y_labels=['mean nontargets activity (z scored)'], fig=fig, ax=axs[1],
                               legend_labels=[f'interictal - photostims - $R^2$: {r_value ** 2:.2e}, p = {p_value ** 2:.2e}, $m$ = {slope:.2e}'],
                               show=False)

    axs[1].plot(results.summed_responses['ictal - distal']['targets'], regression_y, color='forestgreen', lw=2)

    # PLOTTING OPTIONS
    axs[0].grid(True)
    axs[1].grid(True)
    # axs[0].set_ylim([-15, 15])
    # axs[1].set_ylim([-15, 15])
    # axs[0].set_xlim([-7, 7])
    # axs[1].set_xlim([-7, 7])
    fig.suptitle('Total z-scored (to baseline) responses for across all exps', wrap=True)
    fig.tight_layout(pad=0.6)
    fig.show()



# %%

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()
results.summed_responses['ictal - proximal'] = {}
results.summed_responses['ictal - distal'] = {}

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,)
                                # run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
def run__ictal_nontargets_responses_processing(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.PhotostimResponsesNonTargets = calculate__summed_responses_nontargets(expobj=kwargs['expobj'])
    expobj.PhotostimResponsesSLMTargets = calculate__summed_responses_targets(expobj=kwargs['expobj'])
    expobj.save()


run__ictal_nontargets_responses_processing()


__collect__summed_activity_vs_targets_activity(results=results, rerun = 1)

__plot__ictal_nontargets_vs_targets_activity(results=results)




# %%




