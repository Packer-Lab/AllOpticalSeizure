import sys

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os
from typing import Union, List

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

# %%

class PhotostimResponsesNonTargetsResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesNonTargets.pkl'

    def __init__(self):
        super().__init__()

        self.summed_responses = {}  #: dictionary of baseline and interictal summed responses of targets and nontargets across experiments
        self.lin_reg_summed_responses = {}  #: dictionary of baseline and interictal linear regression metrics for relating total targets responses and nontargets responses across experiments
        self.avg_responders_num = None  #: average num responders, for pairedmatched experiments between baseline pre4ap and interictal
        self.avg_baseline_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal - collected for cell ids that were sig responders in baseline
        self.avg_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal
        self.sig_units_baseline = {}  #: dictionary of sig responder units for each baseline exp trial
        self.sig_units_interictal = {}  #: dictionary of sig responder units for the interictal condition
        self.baseline_responses: pd.DataFrame = pd.DataFrame(
            {})  #: dataframe of containing individual cell responses, distance to target, distance to sz
        self.interictal_responses: pd.DataFrame = pd.DataFrame(
            {})  #: dataframe of containing individual cell responses, distance to target, distance to sz
        self.binned_distance_vs_responses: dict = {}
        self.binned_distance_vs_responses_shuffled = {}
        self.binned_distance_vs_responses_interictal: dict = {}
        self.binned_distance_vs_responses_shuffled_interictal = {}

    def __repr__(self):
        return f"PhotostimResponsesNonTargetsResults <- Results Analysis Object"

    @property
    def pre4ap_idxs(self):
        # pre4ap
        idxs = np.where(self.baseline_responses['stim_group'] == 'baseline')[0]
        # idxs_2 = np.where(self.responses['distance target'][idxs] < 300)[0]
        return idxs

    @property
    def pre4ap300(self):
        # pre4ap
        idxs = np.where(self.baseline_responses['stim_group'] == 'baseline')[0]
        idxs_2 = np.where(self.baseline_responses['distance target'][idxs] < 300)[0]
        return idxs_2

    @property
    def post4ap_idxs(self):
        # ictal
        idx = np.where(self.interictal_responses['stim_group'] == 'interictal')[0]
        # idxs_2 = np.where(self.interictal_responses['distance target'][idx] < 300)[0]
        return idx

    def collect_nontargets_stim_responses(results):
        """collecting responses of nontargets relative to various variables (see dataframe below)."""
        from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
            PhotostimResponsesQuantificationNonTargets

        df = pd.DataFrame(
            columns=['expID', 'expID_cell', 'stim_idx', 'stim_group', 'photostim response', 'z score response',
                     'distance target', 'distance sz', 'influence response', 'new influence response',
                     'fakestim response',
                     'sz distance group'])

        # PRE 4AP TRIALS
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, set_cache=0,
                                        skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def _collect_responses_distancetarget(**kwargs):
            expobj: alloptical = kwargs['expobj']
            # exp_df_ = pd.DataFrame(columns=['expID_cell', 'stim_idx','z score response', 'distance target', 'distance sz'])

            z_scored_response = expobj.PhotostimResponsesNonTargets.adata.layers['nontargets responses z scored']
            photostim_response = expobj.PhotostimResponsesNonTargets.adata.X
            fakestim_response = expobj.PhotostimResponsesNonTargets.adata.layers['nontargets fakestim_responses']
            distance_targets = expobj.PhotostimResponsesNonTargets.adata.obs['distance to nearest target (um)']
            mean_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['mean_nontargets_responses']
            std_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['std_nontargets_responses']

            ## Chettih - influence metric type analysis of nontargets response:
            print('\-calculating Chettih style influence....')
            influence = np.zeros_like(photostim_response)
            for i, cell_i in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs.index):
                cell_i = int(cell_i)
                mean_inf = expobj.PhotostimResponsesNonTargets.adata.X[i, :] - np.mean(fakestim_response[i, :])
                inf_norm = mean_inf / np.std(mean_inf, ddof=1)
                influence[i, :] = inf_norm

            ## a new influence metric analysis of nontargets response - that leverages the trial-by-trial population wide changes in activity:
            print('\-calculating new z score style influence response....')
            new_influence = np.zeros_like(photostim_response)
            for i, cell_i in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs.index):
                cell_i = int(cell_i)

                new_inf = (expobj.PhotostimResponsesNonTargets.adata.X[i,
                           :] - mean_nontargets_responses) / std_nontargets_responses
                inf_norm = new_inf / np.std(new_inf,
                                            ddof=1)  # not sure yet if i need to be normalizing to the total variability of this or not???
                new_influence[i, :] = new_inf

            collect_df = []
            print('\-collecting datapoints....')
            for i, cell in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs['original_index']):
                for stim_idx in np.where(expobj.PhotostimResponsesNonTargets.adata.var['stim_group'] == 'baseline')[0]:
                    _cell_df = pd.DataFrame({
                        'expID': f'{expobj.t_series_name}',
                        'expID_cell': f'{expobj.t_series_name}_{cell}',
                        'stim_idx': stim_idx,
                        'stim_group': 'baseline',
                        'photostim response': photostim_response[i, stim_idx],
                        'fakestim response': fakestim_response[i, stim_idx],
                        'z score response': z_scored_response[i, stim_idx],
                        'influence response': influence[i, stim_idx],
                        'new influence response': new_influence[i, stim_idx],
                        'distance target': distance_targets[int(i)],
                        'distance sz': None,
                        'sz distance group': None,
                    }, index=[expobj.t_series_name])

                    collect_df.append(_cell_df)

            exp_df_ = pd.concat(collect_df)

            return exp_df_

        # func_collector = _collect_responses_distancetarget()
        # for exp_df in func_collector:
        #     df = pd.concat([df, exp_df])
        #
        # results.baseline_responses = df
        # results.save_results()

        # POST 4AP TRIALS - interictal todo need to run collecting new influence response
        @Utils.run_for_loop_across_exps(run_post4ap_trials=True, set_cache=0,
                                        skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def _collect_responses_distancetarget_distancesz_interictal(**kwrags):
            expobj: Post4ap = kwrags['expobj']
            # exp_df_ = pd.DataFrame(columns=['expID_cell', 'stim_idx', 'z score response', 'distance target', 'distance sz'])

            # find overlapping cells across processed dataset - actually not sure why there isn't 100% overlap but c'st la vie
            _responses_idx = [idx for idx, cell in
                              enumerate(tuple(expobj.PhotostimResponsesNonTargets.adata.obs['original_index'])) if
                              cell in tuple(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'])]
            # _spatial_idx = [idx for idx, cell in enumerate(tuple(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'])) if cell in tuple(expobj.PhotostimResponsesNonTargets.adata.obs['original_index'])]

            assert tuple(expobj.PhotostimResponsesNonTargets.adata.var_names) == tuple(
                expobj.NonTargetsSzInvasionSpatial.adata.var_names), 'mismatching stim indexes in photostim nontargets responses and distance to sz measured for cells'

            interictal_stim_idxs = np.where(expobj.PhotostimResponsesNonTargets.adata.var['stim_group'] == 'interictal')[0]


            z_scored_response = expobj.PhotostimResponsesNonTargets.adata.layers['nontargets responses z scored (to baseline)']
            photostim_response = expobj.PhotostimResponsesNonTargets.adata.X
            sz_distance_groups = expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location']
            mean_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['mean_nontargets_responses']
            std_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['std_nontargets_responses']


            ## a new influence metric analysis of nontargets response - that leverages the widescale changes in variability:
            print('\-calculating new z score style influence response....')
            new_influence = np.zeros_like(photostim_response)
            for i, cell_i in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs.index):
                cell_i = int(cell_i)

                new_inf = (expobj.PhotostimResponsesNonTargets.adata.X[i,
                           :] - mean_nontargets_responses) / std_nontargets_responses
                inf_norm = new_inf / np.std(new_inf,
                                            ddof=1)  # not sure yet if i need to be normalizing to the total variability of this or not???
                new_influence[i, :] = new_inf

            collect_df = []
            for i in _responses_idx:
                for stim_idx in interictal_stim_idxs:
                    stim_idx = int(stim_idx)
                    cell = expobj.PhotostimResponsesNonTargets.adata.obs['original_index'][i]

                    # _spatial_idx = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'] == cell].index[0]
                    _sz_idx = list(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']).index(cell)
                    _target_distance_idx = list(expobj.PhotostimResponsesNonTargets.adata.obs['original_index']).index(
                        cell)

                    if not np.isnan(expobj.NonTargetsSzInvasionSpatial.adata.X[int(_sz_idx), stim_idx]) and \
                            expobj.NonTargetsSzInvasionSpatial.adata.X[int(_sz_idx), stim_idx] > 0:
                        _cell_df = pd.DataFrame({
                            'expID': f'{expobj.t_series_name}',
                            'expID_cell': f'{expobj.t_series_name}_{cell}',
                            'stim_idx': stim_idx,
                            'stim_group':
                                expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][
                                    stim_idx],
                            'photostim response': photostim_response[i, stim_idx],
                            'z score response': z_scored_response[i, stim_idx],
                            'distance target': expobj.PhotostimResponsesNonTargets.adata.obs[
                                'distance to nearest target (um)'][int(_target_distance_idx)],
                            'distance sz': expobj.NonTargetsSzInvasionSpatial.adata.X[
                                int(_sz_idx), stim_idx],
                            'sz distance group': sz_distance_groups[int(_sz_idx), stim_idx]
                        }, index=[expobj.t_series_name])

                        collect_df.append(_cell_df)

                    elif expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][stim_idx] == 'interictal':
                        _cell_df = pd.DataFrame({
                            'expID': f'{expobj.t_series_name}',
                            'expID_cell': f'{expobj.t_series_name}_{cell}',
                            'stim_idx': stim_idx,
                            'stim_group':
                                expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][
                                    stim_idx],
                            'photostim response': photostim_response[i, stim_idx],
                            'z score response': z_scored_response[i, stim_idx],
                            'distance target': expobj.PhotostimResponsesNonTargets.adata.obs[
                                'distance to nearest target (um)'][int(_target_distance_idx)],
                            'distance sz': expobj.NonTargetsSzInvasionSpatial.adata.X[
                                int(_sz_idx), stim_idx],
                            'sz distance group': sz_distance_groups[int(_sz_idx), stim_idx]
                        }, index=[expobj.t_series_name])

                        collect_df.append(_cell_df)

            exp_df_ = pd.concat(collect_df)

            return exp_df_

        func_collector = _collect_responses_distancetarget_distancesz_interictal()
        for exp_df in func_collector:
            df = pd.concat([df, exp_df])

        results.interictal_responses = df
        results.save_results()

        # POST 4AP TRIALS - ictal (split up by proximal and distal to sz) todo need to run collecting new influence response
        @Utils.run_for_loop_across_exps(run_post4ap_trials=True, set_cache=0,
                                        skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def _collect_responses_distancetarget_distancesz_ictal(**kwrags):
            expobj: Post4ap = kwrags['expobj']
            # exp_df_ = pd.DataFrame(columns=['expID_cell', 'stim_idx', 'z score response', 'distance target', 'distance sz'])

            # find overlapping cells across processed dataset - actually not sure why there isn't 100% overlap but c'st la vie
            _responses_idx = [idx for idx, cell in
                              enumerate(tuple(expobj.PhotostimResponsesNonTargets.adata.obs['original_index'])) if
                              cell in tuple(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'])]
            # _spatial_idx = [idx for idx, cell in enumerate(tuple(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'])) if cell in tuple(expobj.PhotostimResponsesNonTargets.adata.obs['original_index'])]

            assert tuple(expobj.PhotostimResponsesNonTargets.adata.var_names) == tuple(
                expobj.NonTargetsSzInvasionSpatial.adata.var_names), 'mismatching stim indexes in photostim nontargets responses and distance to sz measured for cells'

            z_scored_response = expobj.PhotostimResponsesNonTargets.adata.layers[
                'nontargets responses z scored (to baseline)']
            photostim_response = expobj.PhotostimResponsesNonTargets.adata.X
            sz_distance_groups = expobj.NonTargetsSzInvasionSpatial.adata.layers['outsz location']
            mean_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['mean_nontargets_responses']
            std_nontargets_responses = expobj.PhotostimResponsesNonTargets.adata.var['std_nontargets_responses']

            ## Chettih - influence metric type analysis of nontargets response:
            print('\-calculating Chettih style influence....')
            influence = np.zeros_like(photostim_response)
            for i, cell_i in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs.index):
                cell_i = int(cell_i)
                mean_inf = expobj.PhotostimResponsesNonTargets.adata.X[i, :] - np.mean(fakestim_response[i, :])
                inf_norm = mean_inf / np.std(mean_inf, ddof=1)
                influence[i, :] = inf_norm

            ## a new influence metric analysis of nontargets response - that leverages the widescale changes in variability:
            print('\-calculating new z score style influence response....')
            new_influence = np.zeros_like(photostim_response)
            for i, cell_i in enumerate(expobj.PhotostimResponsesNonTargets.adata.obs.index):
                cell_i = int(cell_i)

                new_inf = (expobj.PhotostimResponsesNonTargets.adata.X[i,
                           :] - mean_nontargets_responses) / std_nontargets_responses
                inf_norm = new_inf / np.std(new_inf,
                                            ddof=1)  # not sure yet if i need to be normalizing to the total variability of this or not???
                new_influence[i, :] = new_inf

            collect_df = []
            for i in _responses_idx:
                for stim_idx in np.where(expobj.PhotostimResponsesNonTargets.adata.var['stim_group'] == 'interictal')[0]:
                    stim_idx = int(stim_idx)
                    cell = expobj.PhotostimResponsesNonTargets.adata.obs['original_index'][i]

                    # _spatial_idx = expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'][expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index'] == cell].index[0]
                    _sz_idx = list(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']).index(cell)
                    _target_distance_idx = list(expobj.PhotostimResponsesNonTargets.adata.obs['original_index']).index(
                        cell)

                    if not np.isnan(expobj.NonTargetsSzInvasionSpatial.adata.X[int(_sz_idx), stim_idx]) and \
                            expobj.NonTargetsSzInvasionSpatial.adata.X[int(_sz_idx), stim_idx] > 0:
                        _cell_df = pd.DataFrame({
                            'expID': f'{expobj.t_series_name}',
                            'expID_cell': f'{expobj.t_series_name}_{cell}',
                            'stim_idx': stim_idx,
                            'stim_group':
                                expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][
                                    stim_idx],
                            'photostim response': photostim_response[i, stim_idx],
                            'z score response': z_scored_response[i, stim_idx],
                            'distance target': expobj.PhotostimResponsesNonTargets.adata.obs[
                                'distance to nearest target (um)'][int(_target_distance_idx)],
                            'distance sz': expobj.NonTargetsSzInvasionSpatial.adata.X[
                                int(_sz_idx), stim_idx],
                            'sz distance group': sz_distance_groups[int(_sz_idx), stim_idx]
                        }, index=[expobj.t_series_name])

                        collect_df.append(_cell_df)

                    elif expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][stim_idx] == 'interictal':
                        _cell_df = pd.DataFrame({
                            'expID': f'{expobj.t_series_name}',
                            'expID_cell': f'{expobj.t_series_name}_{cell}',
                            'stim_idx': stim_idx,
                            'stim_group':
                                expobj.PhotostimResponsesNonTargets.adata.var['stim_group'][
                                    stim_idx],
                            'photostim response': photostim_response[i, stim_idx],
                            'z score response': z_scored_response[i, stim_idx],
                            'distance target': expobj.PhotostimResponsesNonTargets.adata.obs[
                                'distance to nearest target (um)'][int(_target_distance_idx)],
                            'distance sz': expobj.NonTargetsSzInvasionSpatial.adata.X[
                                int(_sz_idx), stim_idx],
                            'sz distance group': sz_distance_groups[int(_sz_idx), stim_idx]
                        }, index=[expobj.t_series_name])

                        collect_df.append(_cell_df)

            exp_df_ = pd.concat(collect_df)

            return exp_df_

        func_collector = _collect_responses_distancetarget_distancesz()
        for exp_df in func_collector:
            df = pd.concat([df, exp_df])

        results.interictal_responses = df
        results.save_results()

    def binned_distances_vs_responses_baseline(results, measurement='photostim response'):
        """

        use the specified response measurement argument.
        - bin the dataframe across distance
        - for each distance bin:
            - for each nontarget cell:
                - calculate average measurement response across all stims
            - collect average and std measurement response across all cells in current distance bin

        :param measurement:
        :return:
        """

        # CURRENT SETUP FOR BASELINE RESPONSES ONLY!! ************
        baseline_responses = results.baseline_responses.iloc[results.pre4ap_idxs]

        baseline_responses['shuffled distance'] = 0.0

        # responses sorted by shuffled distance and then binned by 10um bins ########################################
        # baseline_responses_shuffled = baseline_responses.sort_values(by=['distance target'])
        # random_order = np.random.choice(range(baseline_responses.shape[0]), baseline_responses.shape[0], replace=False)

        cells_ = tuple(np.unique(baseline_responses['expID_cell']))
        distances = []
        for cell in cells_:
            idxs = np.where(baseline_responses['expID_cell'] == cell)[0]
            distances.append(baseline_responses['distance target'].iloc[idxs[0]])
            # assert len(np.unique(distance)) == 1
            # distance = baseline_responses['distance target'].iloc[idxs[0]]

        # add shuffled distances
        random_order_distances = np.random.choice(distances, len(distances), replace=False)
        for idx, cell in enumerate(cells_):
            idxs = np.where(baseline_responses['expID_cell'] == cell)[0]
            baseline_responses['shuffled distance'].iloc[idxs] = random_order_distances[idx]

        baseline_responses = baseline_responses.sort_values(by=['shuffled distance'])

        # binning distances - 20um bins
        baseline_responses['shuffled distance binned'] = (baseline_responses['shuffled distance'] // 10) * 10

        # average across distance bins
        # measurement = 'influence response'
        assert measurement in baseline_responses.columns, f'measurement not found in responses df columns:\n\t {baseline_responses.columns}'
        print(f'\- processing measurement: {measurement}')
        distances = np.unique(baseline_responses['shuffled distance binned'])
        avg_binned_responses = []
        std_binned_responses = []
        for bin in distances:
            print(f'\t\- processing distance bin: {bin}')
            _idxs = np.where(baseline_responses['shuffled distance binned'] == bin)

            # average across all stims for each cell
            cells_ = np.unique(baseline_responses.iloc[_idxs]['expID_cell'])
            averages_cells = []
            for cell in cells_:
                _jdxs = np.where(baseline_responses.iloc[_idxs]['expID_cell'] == cell)[0]
                mean_cell_distance_response = np.mean(baseline_responses.iloc[_idxs].iloc[_jdxs][measurement])
                averages_cells.append(mean_cell_distance_response)

            # avg_response = np.mean(baseline_responses[measurement].iloc[_idxs])
            # std_response = np.std(baseline_responses[measurement].iloc[_idxs], ddof=1)

            avg_response = np.mean(averages_cells)
            std_response = np.std(averages_cells, ddof=1)

            avg_binned_responses.append(avg_response)
            std_binned_responses.append(std_response)
        avg_binned_responses = np.asarray(avg_binned_responses)
        std_binned_responses = np.asarray(std_binned_responses)

        if not hasattr(results, 'binned_distance_vs_responses_shuffled'):
            results.binned_distance_vs_responses_shuffled = {}

        results.binned_distance_vs_responses_shuffled[measurement] = {'distances': distances,
                                                                      'avg binned responses': avg_binned_responses,
                                                                      'std binned responses': std_binned_responses}

        results.save_results()

        # binning across distance to target: ############################################################################
        # re-sort by distance to target
        baseline_responses = baseline_responses.sort_values(by=['distance target'])

        # binning distances - 20um bins
        baseline_responses['distance target binned'] = (baseline_responses['distance target'] // 10) * 10

        # average across distance bins
        # measurement = 'influence response'
        assert measurement in baseline_responses.columns, f'measurement not found in responses df columns:\n\t {baseline_responses.columns}'
        print(f'\- processing measurement: {measurement}')
        distances = np.unique(baseline_responses['distance target binned'])
        avg_binned_responses = []
        std_binned_responses = []
        for bin in distances:
            print(f'\t\- processing distance bin: {bin}')
            _idxs = np.where(baseline_responses['distance target binned'] == bin)

            # average across all stims for each cell
            cells_ = np.unique(baseline_responses.iloc[_idxs]['expID_cell'])
            averages_cells = []
            for cell in cells_:
                _jdxs = np.where(baseline_responses.iloc[_idxs]['expID_cell'] == cell)[0]
                mean_cell_distance_response = np.mean(baseline_responses.iloc[_idxs].iloc[_jdxs][measurement])
                averages_cells.append(mean_cell_distance_response)

            # avg_response = np.mean(baseline_responses[measurement].iloc[_idxs])
            # std_response = np.std(baseline_responses[measurement].iloc[_idxs], ddof=1)

            avg_response = np.mean(averages_cells)
            std_response = np.std(averages_cells, ddof=1)

            avg_binned_responses.append(avg_response)
            std_binned_responses.append(std_response)
        avg_binned_responses = np.asarray(avg_binned_responses)
        std_binned_responses = np.asarray(std_binned_responses)

        if not hasattr(results, 'binned_distance_vs_responses'):
            results.binned_distance_vs_responses = {}

        results.binned_distance_vs_responses[measurement] = {'distances': distances,
                                                             'avg binned responses': avg_binned_responses,
                                                             'std binned responses': std_binned_responses}

        results.save_results()

    def binned_distances_vs_responses_interictal(results, measurement='photostim response'):
        """

        use the specified response measurement argument.
        - bin the dataframe across distance
        - for each distance bin:
            - for each nontarget cell:
                - calculate average measurement response across all stims
            - collect average and std measurement response across all cells in current distance bin

        :param measurement:
        :return:
        """

        # CURRENT SETUP FOR interictal RESPONSES ONLY!! ************
        interictal_responses = results.interictal_responses.iloc[results.post4ap_idxs]

        interictal_responses['shuffled distance'] = 0.0

        # responses sorted by shuffled distance and then binned by 10um bins ########################################
        # interictal_responses_shuffled = interictal_responses.sort_values(by=['distance target'])
        # random_order = np.random.choice(range(interictal_responses.shape[0]), interictal_responses.shape[0], replace=False)

        cells_ = tuple(np.unique(interictal_responses['expID_cell']))
        distances = []
        for cell in cells_:
            idxs = np.where(interictal_responses['expID_cell'] == cell)[0]
            distances.append(interictal_responses['distance target'].iloc[idxs[0]])
            # assert len(np.unique(distance)) == 1
            # distance = interictal_responses['distance target'].iloc[idxs[0]]

        # add shuffled distances
        random_order_distances = np.random.choice(distances, len(distances), replace=False)
        for idx, cell in enumerate(cells_):
            idxs = np.where(interictal_responses['expID_cell'] == cell)[0]
            interictal_responses['shuffled distance'].iloc[idxs] = random_order_distances[idx]

        interictal_responses = interictal_responses.sort_values(by=['shuffled distance'])

        # binning distances - 20um bins
        interictal_responses['shuffled distance binned'] = (interictal_responses['shuffled distance'] // 10) * 10

        # average across distance bins
        # measurement = 'influence response'
        assert measurement in interictal_responses.columns, f'measurement not found in responses df columns:\n\t {interictal_responses.columns}'
        print(f'\- processing measurement: {measurement}')
        distances = np.unique(interictal_responses['shuffled distance binned'])
        avg_binned_responses = []
        std_binned_responses = []
        for bin in distances:
            print(f'\t\- processing distance bin: {bin}')
            _idxs = np.where(interictal_responses['shuffled distance binned'] == bin)

            # average across all stims for each cell
            cells_ = np.unique(interictal_responses.iloc[_idxs]['expID_cell'])
            averages_cells = []
            for cell in cells_:
                _jdxs = np.where(interictal_responses.iloc[_idxs]['expID_cell'] == cell)[0]
                mean_cell_distance_response = np.nanmean(interictal_responses.iloc[_idxs].iloc[_jdxs][measurement])
                averages_cells.append(mean_cell_distance_response)

            # avg_response = np.nanmean(interictal_responses[measurement].iloc[_idxs])
            # std_response = np.nanstd(interictal_responses[measurement].iloc[_idxs], ddof=1)
            if len(averages_cells) > 0:
                avg_response = np.nanmean(averages_cells)
                std_response = np.nanstd(averages_cells, ddof=1)

                avg_binned_responses.append(avg_response)
                std_binned_responses.append(std_response)
        avg_binned_responses = np.asarray(avg_binned_responses)
        std_binned_responses = np.asarray(std_binned_responses)

        if not hasattr(results, 'binned_distance_vs_responses_shuffled_interictal'):
            results.binned_distance_vs_responses_shuffled_interictal = {}

        results.binned_distance_vs_responses_shuffled_interictal[measurement] = {'distances': distances,
                                                                                 'avg binned responses': avg_binned_responses,
                                                                                 'std binned responses': std_binned_responses}

        results.save_results()

        # binning across distance to target: ############################################################################
        # re-sort by distance to target
        interictal_responses = interictal_responses.sort_values(by=['distance target'])

        # binning distances - 20um bins
        interictal_responses['distance target binned'] = (interictal_responses['distance target'] // 10) * 10

        # average across distance bins
        # measurement = 'influence response'
        assert measurement in interictal_responses.columns, f'measurement not found in responses df columns:\n\t {interictal_responses.columns}'
        print(f'\- processing measurement: {measurement}')
        distances = np.unique(interictal_responses['distance target binned'])
        avg_binned_responses = []
        std_binned_responses = []
        for bin in distances:
            print(f'\t\- processing distance bin: {bin}')
            _idxs = np.where(interictal_responses['distance target binned'] == bin)

            # average across all stims for each cell
            cells_ = np.unique(interictal_responses.iloc[_idxs]['expID_cell'])
            averages_cells = []
            for cell in cells_:
                _jdxs = np.where(interictal_responses.iloc[_idxs]['expID_cell'] == cell)[0]
                mean_cell_distance_response = np.nanmean(interictal_responses.iloc[_idxs].iloc[_jdxs][measurement])
                averages_cells.append(mean_cell_distance_response)

            # avg_response = np.nanmean(interictal_responses[measurement].iloc[_idxs])
            # std_response = np.nanstd(interictal_responses[measurement].iloc[_idxs], ddof=1)

            if len(averages_cells) > 0:
                avg_response = np.nanmean(averages_cells)
                std_response = np.nanstd(averages_cells, ddof=1)

                avg_binned_responses.append(avg_response)
                std_binned_responses.append(std_response)

        avg_binned_responses = np.asarray(avg_binned_responses)
        std_binned_responses = np.asarray(std_binned_responses)

        if not hasattr(results, 'binned_distance_vs_responses_interictal'):
            results.binned_distance_vs_responses_interictal = {}

        results.binned_distance_vs_responses_interictal[measurement] = {'distances': distances,
                                                                        'avg binned responses': avg_binned_responses,
                                                                        'std binned responses': std_binned_responses}

        results.save_results()


REMAKE = False
if not os.path.exists(PhotostimResponsesNonTargetsResults.SAVE_PATH) or REMAKE:
    results = PhotostimResponsesNonTargetsResults()
    results.save_results()
