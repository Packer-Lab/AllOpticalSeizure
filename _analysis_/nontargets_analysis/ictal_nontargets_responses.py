"""


Current objectives:
- [ ]  scatter plot: total responses of targets in proximal vs. mean z scored (to baseline) responses in proximal
- [ ]  scatter plot: total responses of targets in distal vs. mean z scored (to baseline) responses of nontargets in distal


"""


import sys
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import PhotostimResponsesNonTargetsResults, \
    PhotostimResponsesQuantificationNonTargets, FakeStimsQuantification

from _exp_metainfo_.exp_metainfo import SAVE_LOC


import os
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _main_.AllOpticalMain import alloptical
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

def __collect__summed_activity_vs_targets_activity(results: PhotostimResponsesNonTargetsResults):
    """collect summed zscored activity of nontargets proximal and distal to sz wavefront"""

    # post4ap - interictal #############################################################################################
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
    def collect_summed_responses_interictal(lin_reg_scores, lin_reg_scores_fakestims, **kwargs):
        """collect z scored (to baseline) summed responses for photostim nontargets, split by proximal and distal groups"""
        expobj: Post4ap = kwargs['expobj']

        summed_responses = pd.DataFrame({'exp': [expobj.t_series_name] * sum(
            [expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'][0]),
                                         'targets': expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'],
                                         'proximal non-targets': expobj.PhotostimResponsesNonTargets.adata.var['nontargets - proximal - mean z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal'],
                                         'distal non-targets': expobj.PhotostimResponsesNonTargets.adata.var['nontargets - distal - mean z score'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'ictal']
                                         })


        # calculating linear regression metrics between summed targets and summed total network for each experiment
        # photostims
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=targets_summed_activity_zsco,
                                                                       y=network_summed_activity_zsco)
        regression_y = slope * targets_summed_activity_zsco + intercept

        summed_responses_zscore = pd.DataFrame({'exp': [expobj.t_series_name] * len(regression_y),
                                                'targets_summed_zscored': targets_summed_activity_zsco,
                                                'all_non-targets_zscored': network_summed_activity_zsco,
                                                'all_non-targets_score_regression': regression_y})

        lin_reg_scores = pd.DataFrame({
            'exp': expobj.t_series_name,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value
        }, index=[expobj.t_series_name])

        # fakestims
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=targets_fakestims_summed_activity_zsco,
                                                                       y=network_fakestims_summed_activity_zsco)
        regression_y_fakestims = slope * targets_fakestims_summed_activity_zsco + intercept

        summed_responses_fakestims_zscore = pd.DataFrame({})
        summed_responses_fakestims_zscore[
            'targets_fakestims_summed_zscored'] = targets_fakestims_summed_activity_zsco
        summed_responses_fakestims_zscore[
            'all_non-targets_fakestims_zscored'] = network_fakestims_summed_activity_zsco
        summed_responses_fakestims_zscore['all_non-targets_fakestims_score_regression'] = regression_y_fakestims

        lin_reg_scores_fakestims = pd.DataFrame({
            'exp': expobj.t_series_name,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value
        }, index=[expobj.t_series_name])

        return summed_responses_zscore, summed_responses_fakestims_zscore, lin_reg_scores, lin_reg_scores_fakestims
        # return expobj.PhotostimResponsesNonTargets.adata.var['summed_response_pos_interictal'], expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets']

    func_collector_interictal = collect_summed_responses_interictal(
        lin_reg_scores=results.lin_reg_summed_responses['baseline'],
        lin_reg_scores_fakestims=results.lin_reg_summed_responses['baseline - fakestims'])

    if func_collector_interictal is not None:

        # summed_responses_interictal = pd.DataFrame({'exp': [], 'targets': [], 'non-targets_pos': [], 'non-targets_neg': [], 'all_non-targets': [],
        #                                           'targets_summed_zscored': [], 'all_non-targets_zscored': [], 'all_non-targets_score_regression': []})

        summed_responses_interictal_zscore = pd.DataFrame({'exp': [],
                                                           'targets_summed_zscored': [],
                                                           'all_non-targets_zscored': [],
                                                           'all_non-targets_score_regression': [], })
        # 'targets_fakestims_summed_zscored': [], 'all_non-targets_fakestims_zscored': [], 'all_non-targets_fakestims_score_regression': []})

        summed_responses_fakestims_interictal_zscore = pd.DataFrame({})
        summed_responses_fakestims_interictal_zscore['targets_fakestims_summed_zscored'] = []
        summed_responses_fakestims_interictal_zscore['all_non-targets_fakestims_zscored'] = []
        summed_responses_fakestims_interictal_zscore['all_non-targets_fakestims_score_regression'] = []

        lin_reg_scores_interictal = pd.DataFrame(
            {'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})
        lin_reg_scores_fakestims_interictal = pd.DataFrame(
            {'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})

        for exp in func_collector_interictal:
            summed_responses_interictal_zscore = pd.concat([summed_responses_interictal_zscore, exp[0]])
            summed_responses_fakestims_interictal_zscore = pd.concat(
                [summed_responses_fakestims_interictal_zscore, exp[1]])
            lin_reg_scores_interictal = pd.concat([lin_reg_scores_interictal, exp[2]])
            lin_reg_scores_fakestims_interictal = pd.concat([lin_reg_scores_fakestims_interictal, exp[3]])

        print('summed responses interictal zscore shape', summed_responses_interictal_zscore.shape)
        print('summed responses fakestims interictal zscore shape',
              summed_responses_fakestims_interictal_zscore.shape)

        results.summed_responses['interictal'] = summed_responses_interictal_zscore
        results.summed_responses['interictal - fakestims'] = summed_responses_fakestims_interictal_zscore
        results.lin_reg_summed_responses['interictal'] = lin_reg_scores_interictal
        results.lin_reg_summed_responses['interictal - fakestims'] = lin_reg_scores_fakestims_interictal
        results.save_results()


# %%

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,
                                run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
def run__ictal_nontargets_responses_processing(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.PhotostimResponsesNonTargets = calculate__summed_responses_nontargets(expobj=kwargs['expobj'])
    expobj.PhotostimResponsesSLMTargets = calculate__summed_responses_targets(expobj=kwargs['expobj'])
    expobj.save()


run__ictal_nontargets_responses_processing()


# %%




