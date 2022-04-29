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


# %% ###### NON TARGETS analysis + plottings

class PhotostimResponsesNonTargetsResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesNonTargets.pkl'

    def __init__(self):
        super().__init__()

        self.summed_responses = None  #: dictionary of baseline and interictal summed responses of targets and nontargets across experiments
        self.lin_reg_summed_responses = None  #: dictionary of baseline and interictal linear regression metrics for relating total targets responses and nontargets responses across experiments
        self.avg_responders_num = None  #: average num responders, for pairedmatched experiments between baseline pre4ap and interictal
        self.avg_baseline_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal - collected for cell ids that were sig responders in baseline
        self.avg_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal
        self.sig_units_baseline = {}  #: dictionary of sig responder units for each baseline exp trial
        self.sig_units_interictal = {}  #: dictionary of sig responder units for the interictal condition


REMAKE = False
if not os.path.exists(PhotostimResponsesNonTargetsResults.SAVE_PATH) or REMAKE:
    results = PhotostimResponsesNonTargetsResults()
    results.save_results()




######

class FakeStimsQuantification(Quantification):
    """class for holding analyses attr for fakestim responses."""

    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    EXCLUDE_TRIALS = ['RL108 t-009',
                      'RL109 t-013'  # these both have negative going targets traces in the fake stims period
                      ]

    def __init__(self, expobj):
        super().__init__(expobj=expobj)
        self.diff_responses_array: np.ndarray = None  #: array of responses calculated for nontargets from fakestims


class PhotostimResponsesQuantificationNonTargets(Quantification):
    """class for quanitying responses of non-targeted cells at photostimulation trials.
    non-targeted cells classified as Suite2p ROIs that were not SLM targets.


    Tasks:
    [x] scatter plot of individual stim trials: response magnitude of targets vs. response magnitude of all (significantly responding) nontargets
                                                                                - (maybe, evoked summed magnitude of pos and neg sig. responders - check previous plots on this to see if there's some insights there....)
        [x] set up code to split post4ap nontargets responses collected data into interictal and ictal
        [x] continue working thorugh workflow to collect traces for pos and neg sig. responders - pre4ap and post4ap
        [x] need to think of a way to normalize results within experiments before collating across experiments
            [x] maybe calcualte Pearson's r for each experiment between baseline and interictal, and compare these values across experiments
            [x] or simple z - scoring of targets and non-targets responses --> think I'll try this first - do for one experiment and plot results
        [x] also might consider measuring activity responses of ALL cells, not just the significantly responding nontargets < -- seems to be a promising avenue

        [x] plan: calculate $R^2$ for each experiment (pre4ap and post4ap interictal), and compare on bar plot
            [x] - could also consider aggregating z scores across all experiments for baseline and interictal stims

    [x] quantifying nontargets responses inside and outside sz boundary - during ictal stims
        - how do you setup the stats tests for nontargets? do you exclude cells that are inside the sz boundary for certain stims?
        [x] alt. approach is to take the significant responders from baseline, and apply those same responders to interictal and ictal:
            - i.e. don't quantify significant responders separately in interictal and ictal
            - collect__sig_responders_responses_type2 <- testing this right now, and then use outputs to create required plots of response magnitudes for baseline, interictal, and in/out of seizure
        [x] collect nontargets traces during seizures and correlate with seizure position
        [x] building code to collect nontagets traces from just out sz cells during sz stims


    """

    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    nontargets_sig_fdr_alpha = 0.25  #: FDR alpha level for finding significant nontargets after running wilcoxon test
    mean_photostim_responses_baseline: List[float] = None
    mean_photostim_responses_interictal: List[float] = None
    mean_photostim_responses_ictal: List[float] = None
    EXCLUDE_TRIALS = ['PS04 t-012',  # no responding cells and also doubled up by PS04 t-017 in any case
                      ]

    REPRESENTATIVE_TRIALS = ['PS07 t-007', 'PS07 t-011']
    TEST_TRIALS = [
                    'RL108 t-009'
                   # 'RL108 t-013'
                    ]

    def __init__(self, results, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)

        # self = expobj.PhotostimResponsesNonTargets  # temp during testing, can remove when running all experiments.
        # if self.responders is None:  # temp during testing, can remove when running all experiments.

        self.diff_responses = None  #: subtraction of stim responses post array with stim responses pre array (direct diff. of post-stim - pre-stim value for each nontarget across all stims)
        self.wilcoxons = None
        self.sig_units = None

        print(f'\- ADDING NEW PhotostimResponsesNonTargets MODULE to expobj: {expobj.t_series_name}')
        self._allopticalAnalysisNontargets(expobj=expobj, results=results)
        results.save_results()
        
        if not hasattr(expobj, 's2p_nontargets'):
            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(plot=False)
            expobj.save()
        self.collect__sig_responders_responses_type1(expobj=expobj)
        self.collect__sig_responders_responses_type2(expobj=expobj, results=results)
        self.create_anndata(expobj=expobj)  # <- still need to test!

        self.fakestims: FakeStimsQuantification = FakeStimsQuantification(expobj=expobj)
        self.fakestims_allopticalAnalysisNontargets(expobj=expobj)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, skip_trials=EXCLUDE_TRIALS,
                                    run_trials=TEST_TRIALS)
    def run__methods(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        # expobj.PhotostimResponsesNonTargets._allopticalAnalysisNontargets(expobj=expobj, results=results)
        # collect traces of statistically significant followers:
        # expobj.PhotostimResponsesNonTargets.collect__sig_responders_responses_type1(expobj=expobj)
        # expobj.PhotostimResponsesNonTargets.collect__sig_responders_responses_type2(expobj=expobj, results=results)
        expobj.save()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, allow_rerun=0, skip_trials=EXCLUDE_TRIALS,)
                                    # run_trials=TEST_TRIALS)
    def run__fakestims_processing(**kwargs):
        expobj: alloptical = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets.fakestims = FakeStimsQuantification(expobj=expobj)
        expobj.PhotostimResponsesNonTargets.fakestims_allopticalAnalysisNontargets(expobj=expobj)
        expobj.PhotostimResponsesNonTargets.add_fakestims_anndata()
        expobj.save()


    def __repr__(self):
        return f"PhotostimResponsesNonTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # # 0.1) CLASSIFY NONTARGETS ACROSS SEIZURE BOUNDARY, INCLUDING MEASURING DISTANCE TO SEIZURE BOUNDARY - refactoring out to _ClassNonTargetsSzInvasionSpatial
    # def _classify_nontargets_szboundary(self, expobj: Post4ap, force_redo = False):
    #     """set a matrix of 1s/0s that classifies nontargets inside seizure boundary.
    #     1: outside sz boundary
    #     0: inside sz boundary
    #
    #     """
    #
    #     if not hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim') or force_redo:
    #         expobj.ExpSeizure._procedure__classifying_sz_boundary(expobj=expobj, cells='nontargets')
    #
    #     assert hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim')
    #     if not hasattr(expobj, 'stimsSzLocations'): expobj.sz_locations_stims()
    #
    #
    # def _calc_min_distance_sz_nontargets(self, expobj: Post4ap):
    #     assert hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim')
    #
    #     distance_to_sz_df = expobj.calcMinDistanceToSz_newer(analyse_cells='s2p nontargets', show_debug_plot=False)
    #
    #     assert distance_to_sz_df.shape == self.adata.shape
    #     # add distance to sz df to anndata - similar to targets distance to sz df
    #     self.adata.add_layer(layer_name='distance_to_sz (pixels)', data=distance_to_sz_df)
    #     # distance_to_sz_arr = np.array(distance_to_sz_df['SLM Targets'])
    #
    #     # convert distances to sz boundary (in pixels) to microns - copied from targets sz invasion spatial
    #     self.adata.add_layer(layer_name= 'distance_to_sz (um)', data=(distance_to_sz_df / expobj.pix_sz_x))
    #
    # def _add_nontargets_sz_boundary_anndata(self):
    #     """add layer to anndata table that splits nontarget cell in or out of sz boundary"""
    #
    #     arr = np.empty_like(self.adata.layers['distance_to_sz (pixels)'])
    #
    #     arr[np.where(self.adata.layers['distance_to_sz (pixels)'] > 0)] = 1
    #     arr[np.where(self.adata.layers['distance_to_sz (pixels)'] < 0)] = 0
    #
    #     self.adata.add_layer(layer_name='in/out sz', data=arr)
    #
    #
    # @staticmethod
    # @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, skip_trials=EXCLUDE_TRIALS, allow_rerun=1)
    # def run__classify_and_measure_nontargets_szboundary(force_redo=False, **kwargs):
    #     expobj: Post4ap = kwargs['expobj']
    #     expobj.PhotostimResponsesNonTargets._classify_nontargets_szboundary(expobj=expobj, force_redo=force_redo)
    #     expobj.PhotostimResponsesNonTargets._calc_min_distance_sz_nontargets(expobj=expobj, force_redo=force_redo)
    #     expobj.PhotostimResponsesNonTargets._add_nontargets_sz_boundary_anndata()
    #     expobj.save()

    # 0) PRIMARY ANALYSIS OF NON-TARGETS IN ALL OPTICAL EXPERIMENTS.

    def _allopticalAnalysisNontargets(self, expobj: Union[alloptical, Post4ap], results: PhotostimResponsesNonTargetsResults):
        if 'pre' in expobj.exptype:
            # todo, deprecate use of self.sig_units in favor of self.responders (do it slowly as you come across each bit of code)
            self.diff_responses, self.wilcoxons, self.sig_units, self.responders = expobj._trialProcessing_nontargets(normalize_to='pre-stim', stims = 'all', fdr_alpha=self.nontargets_sig_fdr_alpha,
                                                                                                                      pre_stim_fr=self.pre_stim_fr, pre_stim_response_frames_window=self.pre_stim_response_frames_window,
                                                                                                                      post_stim_response_frames_window=self.post_stim_response_frames_window)
            # self.sig_units = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1, save=False)  #: array of bool describing statistical significance of responder

            # save statisticaly significant units for baseline condition to results object.
            results.sig_units_baseline[expobj.t_series_name] = {}
            results.sig_units_baseline[expobj.t_series_name]['responders'] = self.responders   # cell IDs
            results.sig_units_baseline[expobj.t_series_name]['pos'] = self.pos_sig_responders  # indexes from .responders
            results.sig_units_baseline[expobj.t_series_name]['neg'] = self.neg_sig_responders  # indexes from .responders


            self.dff_stimtraces = expobj.dff_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dff_avgtraces = expobj.dff_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet
            self.dfstdF_stimtraces = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dfstdF_avgtraces = expobj.dfstdF_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet

        elif 'post' in expobj.exptype:
            expobj: Post4ap = expobj
            # all stims
            self.diff_responses, self.wilcoxons, self.sig_units, self.responders = expobj._trialProcessing_nontargets(normalize_to='pre-stim', stims = 'all', fdr_alpha=self.nontargets_sig_fdr_alpha,
                                                                                                                      pre_stim_fr=self.pre_stim_fr, pre_stim_response_frames_window=self.pre_stim_response_frames_window,
                                                                                                                      post_stim_response_frames_window=self.post_stim_response_frames_window)
            self.dff_stimtraces = expobj.dff_traces_nontargets
            self.dfstdF_stimtraces = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet

            # interictal stims only
            self.diff_responses_interictal, self.wilcoxons_interictal, self.sig_units_interictal, self.responders_interictal = expobj._trialProcessing_nontargets(normalize_to='pre-stim',
                                                                                                                                      stims=expobj.stims_out_sz, fdr_alpha=self.nontargets_sig_fdr_alpha,
                                                                                                                                                                  pre_stim_fr=self.pre_stim_fr,
                                                                                                                                                                  pre_stim_response_frames_window=self.pre_stim_response_frames_window,
                                                                                                                                                                  post_stim_response_frames_window=self.post_stim_response_frames_window
                                                                                                                                                                  )

            # save statisticaly significant units for baseline condition to results object.
            results.sig_units_interictal[expobj.t_series_name] = {}
            results.sig_units_interictal[expobj.t_series_name]['responders'] = self.responders_interictal
            results.sig_units_interictal[expobj.t_series_name]['pos'] = self.pos_sig_responders
            results.sig_units_interictal[expobj.t_series_name]['neg'] = self.neg_sig_responders


            # self.sig_units_interictal = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1, save=False)
            self.dff_stimtraces_interictal = expobj.dff_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dff_avgtraces_interictal = expobj.dff_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet
            self.dfstdF_stimtraces_interictal = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dfstdF_avgtraces_interictal = expobj.dfstdF_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet



            # ICTAL stims only
            self.diff_responses_ictal, self.wilcoxons_ictal, self.sig_units_ictal, self.responders_ictal = expobj._trialProcessing_nontargets(normalize_to='pre-stim',
                                                                                                                       stims=expobj.stims_in_sz, fdr_alpha=0.25,
                                                                                                                                      pre_stim_fr=self.pre_stim_fr,
                                                                                                                                              pre_stim_response_frames_window=self.pre_stim_response_frames_window,
                                                                                                                                              post_stim_response_frames_window=self.post_stim_response_frames_window
                                                                                                                                              )

            self.dff_stimtraces_ictal = expobj.dff_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dfstdF_stimtraces_ictal = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet

            # self.sig_units_ictal = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1)


        # # make figure containing plots showing average responses of nontargets to photostim
        # # save_plot_path = expobj.analysis_save_path[:30] + 'Results_figs/' + save_plot_suffix
        # fig_non_targets_responses(expobj=expobj, plot_subset=False,
        #                           save_fig_suffix=save_plot_suffix) if to_plot else None

        print('\n** FIN. * allopticalAnalysisNontargets * %s %s **** ' % (
            expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        print(
            '-------------------------------------------------------------------------------------------------------------\n\n')

    def fakestims_allopticalAnalysisNontargets(self, expobj: alloptical):
        #### TODO DEVELOPING CODE FOR COLLECTING FAKESTIM TRACES FOR NONTARGETS!!!!!!!

        if 'pre' in expobj.exptype:
            expobj._makeNontargetsStimTracesArray(stim_frames=expobj.fake_stim_start_frames, normalize_to='pre-stim',
                                                  save=False, plot=False)

            pre_stim_fr = self.pre_stim_fr
            pre_stim_response_frames_window = self.pre_stim_response_frames_window
            post_stim_response_frames_window = self.post_stim_response_frames_window

            # create parameters, slices, and subsets for making pre-stim and post-stim arrays to use in stats comparison
            # test_period = expobj.pre_stim_response_window_msec / 1000  # sec
            # expobj.test_frames = int(expobj.fps * test_period)  # test period for stats
            self.fakestims.pre_stim_frames_test = np.s_[pre_stim_fr - pre_stim_response_frames_window: pre_stim_fr]
            stim_end = pre_stim_fr + expobj.stim_duration_frames
            self.fakestims.post_stim_frames_test = np.s_[stim_end: stim_end + post_stim_response_frames_window]

            # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
            # analysis_array = expobj.dff_traces_nontargets  # NOTE: USING dFF TRACES
            analysis_array = expobj.fakestims_dfstdF_traces_nontargets  # NOTE: USING dF/stdF TRACES
            self.fakestims.pre_array = np.mean(analysis_array[:, :, self.fakestims.pre_stim_frames_test], axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
            self.fakestims.post_array = np.mean(analysis_array[:, :, self.fakestims.post_stim_frames_test], axis=1)  # [cells x poststim frames] (avg'd taken over all stims)

            self.fakestims.post_array_responses = np.mean(analysis_array[:, :, self.fakestims.post_stim_frames_test],
                                                  axis=2)  #: response post- stim for all cells, stims: [cells x stims]
            self.fakestims.pre_array_responses = np.mean(analysis_array[:, :, self.fakestims.pre_stim_frames_test],
                                                 axis=2)  #: response post- stim for all cells, stims: [cells x stims]

            self.fakestims.diff_responses_array = self.fakestims.post_array_responses - self.fakestims.pre_array_responses

            self.fakestims.wilcoxons = expobj._runWilcoxonsTest(array1=self.fakestims.pre_array,
                                                        array2=self.fakestims.post_array)  #: wilcoxon  value across all cells: len = # cells

            # fdr_alpa = 0.20
            self.fakestims.sig_units = expobj._sigTestAvgResponse_nontargets(p_vals=self.fakestims.wilcoxons, alpha=self.nontargets_sig_fdr_alpha)
            self.fakestims.sig_responders = [cell for idx, cell in enumerate(expobj.s2p_nontargets_analysis) if self.fakestims.sig_units[idx]]

            # expobj.save() if save else None

            avg_post = np.mean(self.fakestims.post_array_responses, axis=1)
            avg_pre = np.mean(self.fakestims.pre_array_responses, axis=1)

            # PLOT TO TEST HOW RESPONSES ARE BEING STATISTICALLY FILTERED:
            # fig, axs = plt.subplots(figsize=(4, 3.5))
            # [axs.plot(expobj.fakestims_dfstdF_traces_nontargets[cell], color='gray', alpha=0.2) for cell in range(self.fakestims.pre_array.shape[0])]
            # axs.plot(np.mean(expobj.fakestims_dfstdF_traces_nontargets, axis=0), color='black')
            #
            # fig.suptitle(f"{expobj.t_series_name} - nontargets fakestims responses")
            # fig.show()


        print('\n** FIN. * fake stims allopticalAnalysisNontargets * %s %s **** ' % (
            expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        print(
            '-------------------------------------------------------------------------------------------------------------\n\n')

    # 1) CREATE ANNDATA
    def create_anndata(self, expobj: Union[alloptical, Post4ap]):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the photostim resposnes of all non-target ROIs.

        """

        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from SLM targets information
        # obs_meta = pd.DataFrame({'s2p nontarget idx': expobj.s2p_nontargets}, index=range(len(expobj.s2p_nontargets)))

        obs_meta = pd.DataFrame(
            columns=['original_index', 'footprint', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'radius',
                     'aspect_ratio', 'npix_norm', 'skew', 'std'], index=range(len(expobj.s2p_nontargets_analysis)))
        for i, idx in enumerate(obs_meta.index):
                for __column in obs_meta:
                    obs_meta.loc[i, __column] = expobj.stat[i][__column]

        # add statistically significant responder
        if 'pre' in self.expobj_exptype:
            obs_meta['responder_baseline'] = self.sig_units

            obs_meta['positive_responder_baseline'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.pos_sig_responders:
                    obs_meta.loc[idx, 'positive_responder_baseline'] = True

            obs_meta['negative_responder_baseline'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.neg_sig_responders:
                    obs_meta.loc[idx, 'negative_responder_baseline'] = True


        elif 'post' in self.expobj_exptype:
            obs_meta['responder_interictal'] = self.sig_units_interictal
            obs_meta['responder_ictal'] = self.sig_units_ictal

            # interictal
            obs_meta['positive_responder_interictal'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.pos_sig_responders_interictal:
                    obs_meta.loc[idx, 'positive_responder_interictal'] = True

            obs_meta['negative_responder_interictal'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.neg_sig_responders_interictal:
                    obs_meta.loc[idx, 'negative_responder_interictal'] = True

            # ictal
            obs_meta['positive_responder_ictal'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.pos_sig_responders_ictal:
                    obs_meta.loc[idx, 'positive_responder_ictal'] = True

            obs_meta['negative_responder_ictal'] = [False for i in obs_meta.index]
            for idx, val in enumerate(obs_meta.index):
                if idx in self.neg_sig_responders_ictal:
                    obs_meta.loc[idx, 'negative_responder_ictal'] = True


        # build numpy array for multidimensional obs metadata
        obs_m = {'ypix': [],
                 'xpix': []}
        for col in [*obs_m]:
            for i, idx in enumerate(expobj.s2p_nontargets_analysis):
                if idx not in expobj.s2p_nontargets_exclude:
                    obs_m[col].append(expobj.stat[i][col])
            obs_m[col] = np.asarray(obs_m[col])


        # SETUP THE VARIABLES ANNOTATIONS TO USE IN anndata
        # build dataframe for var annot's - based on stim_start_frames
        # var_meta = pd.DataFrame(index=['im_group', 'im_time_secs'], columns=range(expobj.n_frames))
        # for fr_idx in range(expobj.n_frames):
        #     if 'pre' in expobj.exptype:
        #         var_meta.loc['im_group', fr_idx] = 'baseline'
        #     elif 'post' in expobj.exptype:
        #         var_meta.loc['im_group', fr_idx] = 'interictal' if fr_idx in expobj.im_idx_outsz else 'ictal'
        #
        #     var_meta.loc['im_time_secs', fr_idx] = round(fr_idx / expobj.fps, 3)

        var_meta = pd.DataFrame(index=['stim_group', 'im_time_secs', 'stim_start_frame', 'wvfront in sz', 'seizure location'],
                                columns=range(len(expobj.stim_start_frames)))
        for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
            if 'pre' in expobj.exptype:
                var_meta.loc['wvfront in sz', fr_idx] = None
                var_meta.loc['seizure location', fr_idx] = None
                var_meta.loc['stim_group', fr_idx] = 'baseline'
            elif 'post' in expobj.exptype:
                if stim_frame in expobj.stimsWithSzWavefront:
                    var_meta.loc['wvfront in sz', fr_idx] = True
                    var_meta.loc['seizure location', fr_idx] = (
                        expobj.stimsSzLocations.coord1[stim_frame], expobj.stimsSzLocations.coord2[stim_frame])
                else:
                    var_meta.loc['wvfront in sz', fr_idx] = False
                    var_meta.loc['seizure location', fr_idx] = None
                var_meta.loc['stim_group', fr_idx] = 'ictal' if fr_idx in expobj.stims_in_sz else 'interictal'
            var_meta.loc['stim_start_frame', fr_idx] = stim_frame
            var_meta.loc['im_time_secs', fr_idx] = stim_frame / expobj.fps


        # SET PRIMARY DATA
        assert hasattr(expobj,
                       'PhotostimResponsesSLMTargets'), 'no photostim responses found to use to create anndata base.'
        print(f"\t\----- CREATING annotated data object using AnnData:")
        # create anndata object
        photostim_responses_adata = AnnotatedData2(X=self.diff_responses, obs=obs_meta, var=var_meta.T, obsm=obs_m,
                                                   data_label='nontargets dFF responses')

        print(f"Created: {photostim_responses_adata}")
        self.adata = photostim_responses_adata


    @staticmethod
    def run__create_anndata(rerun=1):
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=rerun, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def _run__create_anndata(**kwargs):
            expobj: Union[alloptical, Post4ap]  = kwargs['expobj']
            expobj.PhotostimResponsesNonTargets.create_anndata(expobj=expobj)
            expobj.save()

        _run__create_anndata()


    def fix_anndata(self, expobj):
        """function to use for fixing anndata object."""
        new_var = pd.Series(name='stim_group', index=self.adata.var.index, dtype='str')

        for fr_idx in self.adata.var.index:
            if 'pre' in self.expobj_exptype:
                new_var[fr_idx] = 'baseline'
            elif 'post' in self.expobj_exptype:
                new_var[fr_idx] = 'interictal' if expobj.stim_start_frames[
                                                      int(fr_idx)] in expobj.stims_out_sz else 'ictal'

        assert new_var.name in self.adata.var_keys()
        self.adata.var[new_var.name] = new_var

    @staticmethod
    def run__fix_anndata(rerun=0):
        @Utils.run_for_loop_across_exps(run_post4ap_trials=1, allow_rerun=rerun, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def __fix_anndata(**kwargs):
            expobj = kwargs['expobj']
            expobj.PhotostimResponsesNonTargets.fix_anndata(expobj=expobj)
            expobj.save()
        __fix_anndata()

    def add_fakestims_anndata(self):
        assert 'pre' in self.expobj_exptype, 'fakestim nontargets responses currenty only available for pre4ap baseline trials'
        print(f"\- adding fakestims responses to anndata array.")
        fakestim_responses = self.fakestims.diff_responses_array
        self.adata.add_layer(layer_name='nontargets fakestim_responses', data=fakestim_responses)


    # 2) COLLECT pos/neg sig. responders traces and responses

    @property
    def pos_sig_responders_idx(self):
        return np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) > 0)[0]
    @property
    def neg_sig_responders_idx(self):
        return np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) < 0)[0]


    @property
    def pos_sig_responders(self):
        return np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) > 0)[0]
    @property
    def neg_sig_responders(self):
        return np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) < 0)[0]


    @property
    def pos_sig_responders_interictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) > 0)[0]

    @property
    def neg_sig_responders_interictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) < 0)[0]


    @property
    def pos_sig_responders_ictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) > 0)[0]

    @property
    def neg_sig_responders_ictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) < 0)[0]

    def collect__sig_responders_responses_type1(self, expobj: Union[alloptical, Post4ap]):
        """
        Collect responses traces of statistically significant positive and negative photostimulation timed followers. Also collect response magnitude of all pos. and neg. responders for all stims.

        type 1: collect response traces from responders that are statistically significant within each individual condition.

        """
        if 'pre' in expobj.exptype:

            # @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, allow_rerun=1)
            # def pre4ap__sig_responders_traces(**kwargs):
            # expobj = kwargs['expobj']
            # self = expobj.PhotostimResponsesNonTargets

            ### BASELINE (PRE-4AP) GROUP
            # pre4ap_possig_responders_avgtrace = []
            # pre4ap_negsig_responders_avgtrace = []

            self.pre4ap_possig_responders_responses = self.diff_responses[self.sig_units][self.pos_sig_responders]  #: response magnitude for all pos responders for all stims
            self.pre4ap_negsig_responders_responses = self.diff_responses[self.sig_units][self.neg_sig_responders]  #: response magnitude for all neg responders for all stims


            self.pre4ap_possig_responders_traces = self.dfstdF_stimtraces[self.sig_units][np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) > 0)[0]]
            self.pre4ap_negsig_responders_traces = self.dfstdF_stimtraces[self.sig_units][np.where(np.nanmean(self.diff_responses[self.sig_units, :], axis=1) < 0)[0]]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D

            # pre4ap_possig_responders_avgtrace.append(pre4ap_possig_responders_avgtrace_)
            # pre4ap_negsig_responders_avgtrace.append(pre4ap_negsig_responders_avgtrace__)
            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = self.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = self.post_stim_fr  # setting the post_stim array collection period again hard



            # positive responders
            pre4ap_possig_responders_avgtraces = np.mean(self.pre4ap_possig_responders_traces, axis=1)
            data_traces = []
            for trace in pre4ap_possig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.pre4ap_possig_responders_avgtraces_baseline = np.array(data_traces)
            print(f"shape of pre4ap_possig_responders trace array {self.pre4ap_possig_responders_avgtraces_baseline.shape} [2.2-1]")
            # print('stop here... [5.5-1]')


            # negative responders
            pre4ap_negsig_responders_avgtraces = np.mean(self.pre4ap_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in pre4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.pre4ap_negsig_responders_avgtraces_baseline = np.array(data_traces);
            print(f"shape of pre4ap_negsig_responders trace array {self.pre4ap_negsig_responders_avgtraces_baseline.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            self.pre4ap_num_pos = self.pre4ap_possig_responders_avgtraces_baseline.shape[0]
            self.pre4ap_num_neg = self.pre4ap_negsig_responders_avgtraces_baseline.shape[0]



        elif 'post' in expobj.exptype:

            # @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)
            # def post4ap__sig_responders_traces(**kwargs):
            # expobj = kwargs['expobj']
            # self = expobj.PhotostimResponsesNonTargets

            ### INTERICTAL GROUP


            self.post4ap_possig_responders_responses_interictal = self.diff_responses_interictal[self.sig_units_interictal][self.pos_sig_responders_interictal]  #: response magnitude for all pos responders for all stims
            self.post4ap_negsig_responders_responses_interictal = self.diff_responses_interictal[self.sig_units_interictal][self.neg_sig_responders_interictal]  #: response magnitude for all neg responders for all stims


            self.post4ap_possig_responders_traces = self.dfstdF_stimtraces_interictal[self.sig_units_interictal][np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) > 0)[0]]
            self.post4ap_negsig_responders_traces = self.dfstdF_stimtraces_interictal[self.sig_units_interictal][np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) < 0)[0]]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D

            # post4ap_possig_responders_avgtrace.append(post4ap_possig_responders_avgtrace_)
            # post4ap_negsig_responders_avgtrace.append(post4ap_negsig_responders_avgtrace__)
            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = self.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = self.post_stim_fr  # setting the post_stim array collection period again hard



            # positive responders
            post4ap_possig_responders_avgtraces = np.mean(self.post4ap_possig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_possig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_possig_responders_avgtraces_interictal = np.array(data_traces)
            print(f"shape of post4ap_possig_responders trace array {self.post4ap_possig_responders_avgtraces_interictal.shape} [2.2-1]")


            # negative responders
            post4ap_negsig_responders_avgtraces = np.mean(self.post4ap_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_negsig_responders_avgtraces_interictal = np.array(data_traces)
            print(f"shape of post4ap_negsig_responders trace array {self.post4ap_negsig_responders_avgtraces_interictal.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            self.post4ap_num_pos_interictal = self.post4ap_possig_responders_avgtraces_interictal.shape[0]
            self.post4ap_num_neg_interictal = self.post4ap_negsig_responders_avgtraces_interictal.shape[0]



            ### ICTAL GROUP - OUTSIDE SEIZURE BOUNDARY
            # - probably need to test if bugging up....
            self.post4ap_possig_responders_responses_ictal = self.diff_responses_ictal[self.sig_units_ictal][self.pos_sig_responders_ictal]  #: response magnitude for all pos responders for all stims
            self.post4ap_negsig_responders_responses_ictal = self.diff_responses_ictal[self.sig_units_ictal][self.neg_sig_responders_ictal]  #: response magnitude for all neg responders for all stims


            self.post4ap_possig_responders_traces = self.dfstdF_stimtraces_ictal[self.sig_units_ictal][np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) > 0)[0]]
            self.post4ap_negsig_responders_traces = self.dfstdF_stimtraces_ictal[self.sig_units_ictal][np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) < 0)[0]]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D

            # post4ap_possig_responders_avgtrace.append(post4ap_possig_responders_avgtrace_)
            # post4ap_negsig_responders_avgtrace.append(post4ap_negsig_responders_avgtrace__)
            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = self.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = self.post_stim_fr  # setting the post_stim array collection period again hard



            # positive responders
            post4ap_possig_responders_avgtraces = np.mean(self.post4ap_possig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_possig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_possig_responders_avgtraces_ictal = np.array(data_traces)
            print(f"shape of post4ap_possig_responders trace array {self.post4ap_possig_responders_avgtraces_ictal.shape} [2.2-1]")


            # negative responders
            post4ap_negsig_responders_avgtraces = np.mean(self.post4ap_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_negsig_responders_avgtraces_ictal = np.array(data_traces)
            print(f"shape of post4ap_negsig_responders trace array {self.post4ap_negsig_responders_avgtraces_ictal.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            self.post4ap_num_pos_ictal = self.post4ap_possig_responders_avgtraces_ictal.shape[0]
            self.post4ap_num_neg_ictal = self.post4ap_negsig_responders_avgtraces_ictal.shape[0]



    def collect__sig_responders_responses_type2(self, expobj: Union[alloptical, Post4ap], results: PhotostimResponsesNonTargetsResults):
        """
        Collect responses traces of statistically significant positive and negative photostimulation timed followers. Also collect response magnitude of all pos. and neg. responders for all stims.

        type 2: collect across conditions, but using responders that are statistically significant in baseline.
        """

        # todo troubleshoot RL109 t018 post4ap (not an issue for pre4ap) issue with the pre-stim part of the average trace not existing on the plot (goes straight into the zero'd photostim part...)

        print(f"\--- Collecting sig. responders responses (type2) ")

        if 'pre' in expobj.exptype:

            # @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, allow_rerun=1)
            # def pre4ap__sig_responders_traces(**kwargs):
            # expobj = kwargs['expobj']
            # self = expobj.PhotostimResponsesNonTargets

            ### BASELINE (PRE-4AP) GROUP
            # pre4ap_possig_responders_avgtrace = []
            # pre4ap_negsig_responders_avgtrace = []

            pos_responders = np.array(self.responders)[results.sig_units_baseline[expobj.t_series_name]['pos']]
            neg_responders = np.array(self.responders)[results.sig_units_baseline[expobj.t_series_name]['neg']]

            pos_responders_idx = [idx for idx, cell in enumerate(expobj.s2p_nontargets_analysis) if cell in pos_responders]
            neg_responders_idx = [idx for idx, cell in enumerate(expobj.s2p_nontargets_analysis) if cell in neg_responders]
            
            self.pre4ap_possig_responders_responses = self.diff_responses[pos_responders_idx]  #: response magnitude for all pos responders for all stims
            self.pre4ap_negsig_responders_responses = self.diff_responses[neg_responders_idx]  #: response magnitude for all neg responders for all stims


            self.pre4ap_possig_responders_traces = self.dfstdF_stimtraces[pos_responders_idx]
            self.pre4ap_negsig_responders_traces = self.dfstdF_stimtraces[neg_responders_idx]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D

            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = self.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = self.post_stim_fr  # setting the post_stim array collection period again hard



            # positive responders
            pre4ap_possig_responders_avgtraces = np.mean(self.pre4ap_possig_responders_traces, axis=1)
            data_traces = []
            for trace in pre4ap_possig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.pre4ap_possig_responders_avgtraces_baseline = np.array(data_traces)
            print(f"shape of pre4ap_possig_responders trace array {self.pre4ap_possig_responders_avgtraces_baseline.shape} [2.2-1]")
            # print('stop here... [5.5-1]')


            # negative responders
            pre4ap_negsig_responders_avgtraces = np.mean(self.pre4ap_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in pre4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.pre4ap_negsig_responders_avgtraces_baseline = np.array(data_traces)
            print(f"shape of pre4ap_negsig_responders trace array {self.pre4ap_negsig_responders_avgtraces_baseline.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            pre4ap_num_pos = self.pre4ap_possig_responders_avgtraces_baseline.shape[0]
            pre4ap_num_neg = self.pre4ap_negsig_responders_avgtraces_baseline.shape[0]
            assert pre4ap_num_pos == len(pos_responders) == len(self.pos_sig_responders) == len(self.pos_sig_responders_idx)
            assert pre4ap_num_neg == len(neg_responders) == len(self.neg_sig_responders) == len(self.neg_sig_responders_idx)



        elif 'post' in expobj.exptype:

            from _exp_metainfo_.exp_metainfo import ExpMetainfo
            pre4ap_tseries_name = ExpMetainfo.alloptical.find_matched_trial(post4ap_trial_name=expobj.t_series_name)
            pre4ap: alloptical = Utils.import_expobj(exp_prep=pre4ap_tseries_name)

            # assert expobj.s2p_nontargets_analysis == pre4ap.s2p_nontargets_analysis, 'mismatched nontargets analysis cells in post4ap vs. pre4ap matched trials.'

            #   there are different cells that are being excluded in the analysis... need to add excluded cells from pre4ap to
            #   post4ap batch, and then continue from there (i think..?) - actually i dont think that this is necessary. because
            #   you can match the cells ID from pre directly to the indexes of cells that were collected for all optical analysis in post4ap.
            # _s2p_nontargets_analysis = [cell for cell in expobj.s2p_nontargets_analysis if cell not in pre4ap.s2p_nontargets_exclude]  # exclude cells that are also excluded in pre4ap nontargets analysis

            # pick responders (pos and negative) that were determined from baseline condition
            baseline_pos_responders = np.array(results.sig_units_baseline[pre4ap_tseries_name]['responders'])[results.sig_units_baseline[pre4ap_tseries_name]['pos']]
            baseline_neg_responders = np.array(results.sig_units_baseline[pre4ap_tseries_name]['responders'])[results.sig_units_baseline[pre4ap_tseries_name]['neg']]

            # baseline_pos_responders = np.array(self.responders)[results.sig_units_baseline[pre4ap_tseries_name]['pos']]
            # baseline_neg_responders = np.array(self.responders)[results.sig_units_baseline[pre4ap_tseries_name]['neg']]


            baseline_pos_responders_idx = [idx for idx, cell in enumerate(expobj.s2p_nontargets_analysis) if cell in baseline_pos_responders]
            baseline_neg_responders_idx = [idx for idx, cell in enumerate(expobj.s2p_nontargets_analysis) if cell in baseline_neg_responders]


            ### INTERICTAL GROUP
            self.post4ap_baseline_possig_responders_responses_interictal = self.diff_responses_interictal[baseline_pos_responders_idx]  #: response magnitude for all baseline pos responders for all stims
            self.post4ap_baseline_negsig_responders_responses_interictal = self.diff_responses_interictal[baseline_neg_responders_idx]  #: response magnitude for all baseline neg responders for all stims

            self.post4ap_baseline_possig_responders_traces = self.dfstdF_stimtraces_interictal[baseline_pos_responders_idx]
            self.post4ap_baseline_negsig_responders_traces = self.dfstdF_stimtraces_interictal[baseline_neg_responders_idx]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D
            # post4ap_possig_responders_avgtrace.append(post4ap_possig_responders_avgtrace_)
            # post4ap_negsig_responders_avgtrace.append(post4ap_negsig_responders_avgtrace__)
            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = self.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = self.post_stim_fr  # setting the post_stim array collection period again hard



            # positive responders
            post4ap_possig_responders_avgtraces = np.mean(self.post4ap_baseline_possig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_possig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_baseline_possig_responders_avgtraces_interictal = np.array(data_traces)
            print(f"shape of post4ap_possig_responders trace array {self.post4ap_baseline_possig_responders_avgtraces_interictal.shape} [2.2-1]")
            # print('stop here... [5.5-1]')


            # negative responders
            post4ap_negsig_responders_avgtraces = np.mean(self.post4ap_baseline_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_baseline_negsig_responders_avgtraces_interictal = np.array(data_traces)
            print(f"shape of post4ap_negsig_responders trace array {self.post4ap_baseline_negsig_responders_avgtraces_interictal.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            ### ICTAL GROUP - OUTSIDE SEIZURE BOUNDARY
            if not 'in/out sz' in expobj.NonTargetsSzInvasionSpatial.adata.layers:
                expobj.NonTargetsSzInvasionSpatial._add_nontargets_sz_boundary_anndata()
                expobj.save()

            self.post4ap_baseline_possig_responders_responses_ictal = self.diff_responses_ictal[baseline_pos_responders_idx]  #: response magnitude for all baseline pos responders for all stims
            self.post4ap_baseline_negsig_responders_responses_ictal = self.diff_responses_ictal[baseline_neg_responders_idx]  #: response magnitude for all baseline neg responders for all stims

            self.post4ap_baseline_possig_responders_traces = self.dfstdF_stimtraces_ictal[baseline_pos_responders_idx]
            self.post4ap_baseline_negsig_responders_traces = self.dfstdF_stimtraces_ictal[baseline_neg_responders_idx]

            try:
                assert expobj.NonTargetsSzInvasionSpatial.adata.n_obs == len(expobj.s2p_nontargets_analysis), 'mistmatch between spatial cells and nontargets responses analysis cells'
            except AssertionError:
                print('debug here')

            # filter _traces array to create avg traces array from only nontargets that are outside the sz boundary
            post4ap_baseline_possig_responders_avgtraces_ictal = []
            adata_pos_idxs_cells = [(idx, cell) for idx, cell in enumerate(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']) if cell in baseline_pos_responders]  # indexes of cells and cell IDs in baseline_pos_responders from the spatial sz invasion nontargets adata object
            for adata_pos_idx, cell in adata_pos_idxs_cells:
                stim_idxs = np.where(expobj.NonTargetsSzInvasionSpatial.adata.layers['in/out sz'][adata_pos_idx] == 1)[0]  # get all stims where cell is outside of sz.
                if len(stim_idxs) > 0:  # some cells will have no stims for which that cell is outside the seizure.
                    trace_pos_idx = expobj.s2p_nontargets_analysis.index(cell)      # index of the cell of concern in relation to the nontargets all optical responses stim traces analysis
                    # collect all stim traces at these stim idxs
                    traces = []
                    for s_idx in stim_idxs:
                        traces.append(self.dfstdF_stimtraces[trace_pos_idx, s_idx, :])
                    # average these stim traces
                    assert len(traces) > 0, 'no traces to average!'
                    avg_traces = np.mean(traces, axis=0) # this is the average trace for this nontarget cell for out sz, add to overall list:
                    post4ap_baseline_possig_responders_avgtraces_ictal.append(avg_traces)

            data_traces = []
            for trace in post4ap_baseline_possig_responders_avgtraces_ictal:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_baseline_possig_responders_avgtraces_ictal = np.array(data_traces)
            print(f"shape of post4ap_possig_responders trace array (for stims/cells outside sz) {self.post4ap_baseline_possig_responders_avgtraces_interictal.shape} [2.2-1]")


            post4ap_baseline_negsig_responders_avgtraces_ictal = []
            adata_neg_idxs_cells = [(idx, cell) for idx, cell in enumerate(expobj.NonTargetsSzInvasionSpatial.adata.obs['original_index']) if cell in baseline_neg_responders]  # indexes of cells and cell IDs in baseline_neg_responders from the spatial sz invasion nontargets adata object
            for adata_neg_idx, cell in adata_neg_idxs_cells:
                stim_idxs = np.where(expobj.NonTargetsSzInvasionSpatial.adata.layers['in/out sz'][adata_neg_idx] == 1)[0]
                if len(stim_idxs) > 0:  # some cells will have no stims for which that cell is outside the seizure.
                    trace_neg_idx = expobj.s2p_nontargets_analysis.index(cell)      # index of the cell of concern in relation to the nontargets all optical responses stim traces analysis
                    # collect all stim traces at these stim idxs
                    traces = []
                    for s_idx in stim_idxs:
                        traces.append(self.dfstdF_stimtraces[trace_neg_idx, s_idx, :])
                    # average these stim traces
                    assert len(traces) > 0, 'no traces to average!'
                    avg_traces = np.mean(traces, axis=0) # this is the average trace for this nontarget cell for out sz, add to overall list:
                    post4ap_baseline_negsig_responders_avgtraces_ictal.append(avg_traces)

            data_traces = []
            for trace in post4ap_baseline_negsig_responders_avgtraces_ictal:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_baseline_negsig_responders_avgtraces_ictal = np.array(data_traces)
            print(f"shape of post4ap_negsig_responders trace array (for stims/cells outside sz) {self.post4ap_baseline_negsig_responders_avgtraces_interictal.shape} [2.2-1]")






# %%
if __name__ == '__main__':
    # expobj: Post4ap = Utils.import_expobj(exp_prep='RL108 t-013')

    main = PhotostimResponsesQuantificationNonTargets
    results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()


    # running fake stims alloptical analysis for non targets here currently: (apr 28 2022)
    # main.run__methods()
    # main.run__fakestims_processing()

    from _analysis_._ClassPhotostimResponsesAnalysisNonTargets import PhotostimResponsesAnalysisNonTargets
    PhotostimResponsesAnalysisNonTargets.run__plot_sig_responders_traces(plot_baseline_responders=False)

    PhotostimResponsesAnalysisNonTargets.plot__exps_summed_nontargets_vs_summed_targets

    # main.run__create_anndata()

    # main.run__fix_anndata()



    # expobj = Utils.import_expobj(exp_prep='RL108 t-009')




# ARCHIVE
def fig_non_targets_responses(expobj, plot_subset: bool = True, save_fig_suffix=None):
    print('\n----------------------------------------------------------------')
    print('plotting nontargets responses ')
    print('----------------------------------------------------------------')

    if plot_subset:
        selection = np.random.randint(0, expobj.dff_traces_nontargets_avg.shape[0], 100)
    else:
        selection = np.arange(expobj.dff_traces_nontargets_avg.shape[0])

    #### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
    if sum(expobj.sig_units) > 0:
        f = plt.figure(figsize=[25, 10])
        gs = f.add_gridspec(2, 9)
    else:
        f = plt.figure(figsize=[25, 5])
        gs = f.add_gridspec(1, 9)

    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    from _utils_ import alloptical_plotting as aoplot

    a1 = f.add_subplot(gs[0, 0:2])
    x = expobj.dff_traces_nontargets_avg[selection]
    y_label = 'pct. dFF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a1,
                                  show=False,
                                  x_label='Time (seconds)', y_lims=[-50, 200])
    # PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
    a2 = f.add_subplot(gs[0, 2:4])
    x = expobj.dfstdF_traces_nontargets_avg[selection]
    y_label = 'dFstdF (normalized to prestim period)'
    aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                                  title='Average photostim all trials response', y_label=y_label, fig=f, ax=a2,
                                  show=False,
                                  x_label='Time (seconds)', y_lims=[-1, 3])
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/f
    a3 = f.add_subplot(gs[0, 4:6])
    vmin = -1
    vmax = 1
    aoplot.plot_traces_heatmap(arr=expobj.dfstdF_traces_nontargets_avg, expobj=expobj, vmin=vmin, vmax=vmax,
                               stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_nontargets_avg.shape[1]),
                               title='dF/stdF heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a3, show=False)
    # PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/stdf
    a4 = f.add_subplot(gs[0, -3:-1])
    vmin = -100
    vmax = 100
    aoplot.plot_traces_heatmap(arr=expobj.dff_traces_nontargets_avg, expobj=expobj, vmin=vmin, vmax=vmax,
                               stim_on=int(1 * expobj.fps),
                               stim_off=int(1 * expobj.fps + expobj.stim_duration_frames),
                               xlims=(0, expobj.dfstdF_traces_nontargets_avg.shape[1]),
                               title='dF/F heatmap for all nontargets', x_label='Time', cbar=True,
                               fig=f, ax=a4, show=False)
    # bar plot of avg post stim response quantified between responders and non-responders
    a04 = f.add_subplot(gs[0, -1])
    sig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)
    nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
    data = np.asarray([sig_responders_avgresponse, nonsig_responders_avgresponse])
    pplot.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'gray'],
                            y_label='avg dF/stdF', bar=False,
                            text_list=['%s pct' % (np.round(
                                (len(sig_responders_avgresponse) / expobj.post_array_responses.shape[0]), 2) * 100),
                                       '%s pct' % (np.round(
                                           (len(nonsig_responders_avgresponse) / expobj.post_array_responses.shape[0]),
                                           2) * 100)],
                            text_y_pos=1.43, text_shift=1.7, x_tick_labels=['significant', 'non-significant'],
                            ylims=[-2, 3],
                            expand_size_y=1.5, expand_size_x=0.6,
                            fig=f, ax=a04, show=False)

    ## PLOTTING STATISTICALLY SIGNIFICANT RESPONDERS
    if sum(expobj.sig_units) > 0:
        # plot PERI-STIM AVG TRACES of sig nontargets
        a10 = f.add_subplot(gs[1, 0:2])
        x = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a10, show=False,
                                      title='significant responders', y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of nonsig nontargets
        a11 = f.add_subplot(gs[1, 2:4])
        x = expobj.dfstdF_traces_nontargets_avg[~expobj.sig_units]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a11, show=False,
                                      title='non-significant responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. positive responders
        a12 = f.add_subplot(gs[1, 4:6])
        x = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a12, show=False,
                                      title='positive signif. responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # plot PERI-STIM AVG TRACES of sig. negative responders
        a13 = f.add_subplot(gs[1, -3:-1])
        x = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a13, show=False,
                                      title='negative signif. responders',
                                      y_label='dFstdF (normalized to prestim period)',
                                      x_label='Time (seconds)', y_lims=[-1, 3])

        # bar plot of avg post stim response quantified between responders and non-responders
        a14 = f.add_subplot(gs[1, -1])
        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
        data = np.asarray([possig_responders_avgresponse, negsig_responders_avgresponse, nonsig_responders_avgresponse])
        pplot.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells',
                                colors=['green', 'blue', 'gray'],
                                y_label='avg dF/stdF', bar=False,
                                text_list=['%s pct' % (np.round(
                                    (len(possig_responders_avgresponse) / expobj.post_array_responses.shape[0]) * 100,
                                    1)),
                                           '%s pct' % (np.round((len(negsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1)),
                                           '%s pct' % (np.round((len(nonsig_responders_avgresponse) /
                                                                 expobj.post_array_responses.shape[0]) * 100, 1))],
                                text_y_pos=1.43, text_shift=1.2, ylims=[-2, 3],
                                x_tick_labels=['pos. significant', 'neg. significant', 'non-significant'],
                                expand_size_y=1.5, expand_size_x=0.5,
                                fig=f, ax=a14, show=False)

    f.suptitle(
        ('%s %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])))
    f.tight_layout()
    f.show()

    Utils.save_figure(f, save_fig_suffix) if save_fig_suffix is not None else None
    # _path = save_fig_suffix[:[i for i in re.finditer('/', save_fig_suffix)][-1].end()]
    # os.makedirs(_path) if not os.path.exists(_path) else None
    # print('saving figure output to:', save_fig_suffix)
    # plt.savefig(save_fig_suffix)
