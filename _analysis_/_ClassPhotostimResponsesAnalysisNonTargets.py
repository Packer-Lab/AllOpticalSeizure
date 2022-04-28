import sys

from _analysis_._ClassPhotostimResponseQuantificationNonTargets import PhotostimResponsesNonTargetsResults, \
    PhotostimResponsesQuantificationNonTargets

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os
from typing import Union, List

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot


SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# %% ###### NON TARGETS analysis + plottings


REMAKE = False
if not os.path.exists(PhotostimResponsesNonTargetsResults.SAVE_PATH) or REMAKE:
    results = PhotostimResponsesNonTargetsResults()
    results.save_results()


######

class PhotostimResponsesAnalysisNonTargets(PhotostimResponsesQuantificationNonTargets):
    """
    continuation from class: photostim responses quantification nontargets


    [ ] quantify the number of responders that are statistically significant across baseline and interictal (and maybe even out of sz) (as oppossed significant responders within each condition)
    [ ] plotting of significant responders (pos and neg) traces across baseline, interictal and ictal (outsz)

    [i] plotting of significant responders (pos and neg) traces from baseline, during baseline, interictal and ictal (outsz) states


    [ ] fake-sham trials in the analysis.
        [ ] create fake sham stim frames - halfway in between each stim trial
        [ ] collect photostim targets fake sham responses
            [ ] plot peri-stim avg
            [ ] plot response magnitude


        [ ] collect nontargets fake sham responses


    """

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj=expobj, results=results)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1,
                                    run_post4ap_trials=1,
                                    allow_rerun=0,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,)
                                    # run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
    def run__initPhotostimResponsesAnalysisNonTargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets = PhotostimResponsesAnalysisNonTargets(expobj=expobj)
        expobj.save()


    # 2.1) PLOT - POS AND NEG SIG RESPONDERS TRACES FOR EXPERIMENT
    def plot__sig_responders_traces(self, expobj: Union[alloptical, Post4ap]):

        from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
        from _utils_.alloptical_plotting import plot_periphotostim_avg2

        if 'pre' in self.expobj_exptype:
            pos_avg_traces = [self.pre4ap_possig_responders_avgtraces_baseline]
            neg_avg_traces = [self.pre4ap_negsig_responders_avgtraces_baseline]
        elif 'post' in self.expobj_exptype:
            pos_avg_traces = [self.post4ap_possig_responders_avgtraces_interictal]
            neg_avg_traces = [self.post4ap_negsig_responders_avgtraces_interictal]
        else:
            raise Exception()

        fig, axs = plt.subplots(figsize=(4, 6), nrows=2, ncols=1)

        if len(pos_avg_traces[0]) > 0:
            plot_periphotostim_avg2(dataset=pos_avg_traces, fps=expobj.fps,
                                    legend_labels=[f"pos. cells: {len(pos_avg_traces[0])}"],
                                    colors=['red'], avg_with_std=True,
                                    suptitle=f"{self.expobj_id} - {self.expobj_exptype} - sig. responders",
                                    ylim=[-0.3, 0.8], fig=fig, ax=axs[0],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant positive responders.")
        if len(neg_avg_traces[0]) > 0:
            plot_periphotostim_avg2(dataset=neg_avg_traces, fps=expobj.fps,
                                    legend_labels=[f"neg. cells: {len(neg_avg_traces[0])}"],
                                    colors=['blue'], avg_with_std=True,
                                    title=f"{self.expobj_id} - {self.expobj_exptype} - -ve sig. responders",
                                    ylim=[-0.6, 0.5], fig=fig, ax=axs[1],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant negative responders.")

        fig.show()


    # 2.2) PLOT - POS AND NEG SIG RESPONDERS TRACES FOR EXPERIMENT
    @staticmethod
    def plot__pos_neg_responders_traces(expobj: alloptical, pos_avg_traces=None, neg_avg_traces=None, title=''):
        """
        Plot avg peri-stim traces of input +ve and -ve responders traces.

        :param expobj:
        :param pos_avg_traces:
        :param neg_avg_traces:
        """
        if pos_avg_traces is None:
            pos_avg_traces = [[]]
        if neg_avg_traces is None:
            neg_avg_traces = [[]]
        from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
        from _utils_.alloptical_plotting import plot_periphotostim_avg2

        fig, axs = plt.subplots(figsize=(4, 6), nrows=2, ncols=1)

        if len(pos_avg_traces[0]) > 0:
            plot_periphotostim_avg2(dataset=pos_avg_traces, fps=expobj.fps,
                                    legend_labels=[f"pos. cells: {len(pos_avg_traces[0])}"],
                                    colors=['red'], avg_with_std=True,
                                    suptitle=f"{expobj.t_series_name} - {expobj.exptype} - +ve sig. responders {title}",
                                    ylim=[-0.3, 0.8], fig=fig, ax=axs[0],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant positive responders.")
        if len(neg_avg_traces[0]) > 0:
            plot_periphotostim_avg2(dataset=neg_avg_traces, fps=expobj.fps,
                                    legend_labels=[f"neg. cells: {len(neg_avg_traces[0])}"],
                                    colors=['blue'], avg_with_std=True,
                                    title=f"{expobj.t_series_name} - {expobj.exptype} - -ve sig. responders {title}",
                                    ylim=[-0.6, 0.5], fig=fig, ax=axs[1],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant negative responders.")
        fig.show()


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,)
                                    # run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
    def run__plot_sig_responders_traces(plot_baseline_responders=False, **kwargs):
        """
        :param plot_baseline_responders: if True, for post-4ap exp, use the baseline responders' avgtraces for interictal and ictal groups
        :param kwargs:
        """
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        if 'pre' in expobj.exptype:
            pos_avg_traces = [expobj.PhotostimResponsesNonTargets.pre4ap_possig_responders_avgtraces_baseline]
            neg_avg_traces = [expobj.PhotostimResponsesNonTargets.pre4ap_negsig_responders_avgtraces_baseline]

            expobj.PhotostimResponsesNonTargets.plot__pos_neg_responders_traces(expobj=expobj,
                                                                                pos_avg_traces=pos_avg_traces,
                                                                                neg_avg_traces=neg_avg_traces)

        elif 'post' in expobj.exptype:
            # INTERICTAL
            if plot_baseline_responders:
                pos_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_baseline_possig_responders_avgtraces_interictal]  # baseline responders
                neg_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_baseline_negsig_responders_avgtraces_interictal]  # baseline responders

            else:
                pos_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_avgtraces_interictal]
                neg_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_avgtraces_interictal]

            expobj.PhotostimResponsesNonTargets.plot__pos_neg_responders_traces(expobj=expobj,
                                                                                pos_avg_traces=pos_avg_traces,
                                                                                neg_avg_traces=neg_avg_traces,
                                                                                title='interictal datapoints')


            # ICTAL
            if plot_baseline_responders:
                pos_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_baseline_possig_responders_avgtraces_ictal]  # baseline responders
                neg_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_baseline_negsig_responders_avgtraces_ictal]  # baseline responders

            else:
                pos_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_avgtraces_ictal]  # .post4ap_possig_responders_avgtraces_ictal not coded up yet
                neg_avg_traces = [expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_avgtraces_ictal]  # .post4ap_negsig_responders_avgtraces_ictal not coded up yet

            expobj.PhotostimResponsesNonTargets.plot__pos_neg_responders_traces(expobj=expobj,
                                                                                pos_avg_traces=pos_avg_traces,
                                                                                neg_avg_traces=neg_avg_traces,
                                                                                title='outsz datapoints')





    # 2.2) PLOT -- BAR PLOT OF AVG MAGNITUDE OF RESPONSE
    @staticmethod
    def collect__avg_magnitude_response(results: PhotostimResponsesNonTargetsResults, collect_baseline_responders=False):
        """plot bar plot of avg magnitude of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_pos_response(**kwargs):
            """return avg magnitude of positive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_possig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_neg_response(**kwargs):
            """return avg magnitude of negitive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_negsig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_pos_response_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            if collect_baseline_responders:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_baseline_possig_responders_responses_interictal)
            else:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_interictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_neg_response_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            if collect_baseline_responders:
                return np.mean(
                    expobj.PhotostimResponsesNonTargets.post4ap_baseline_negsig_responders_responses_interictal)
            else:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_responses_interictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_pos_response_ictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - ictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            if collect_baseline_responders:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_baseline_possig_responders_responses_ictal)
            else:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_ictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_neg_response_ictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - ictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            if collect_baseline_responders:
                return np.mean(
                    expobj.PhotostimResponsesNonTargets.post4ap_baseline_negsig_responders_responses_ictal)
            else:
                return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_responses_ictal)

        pos_baseline = return__avg_magntiude_pos_response()
        neg_baseline = return__avg_magntiude_neg_response()
        pos_interictal = return__avg_magntiude_pos_response_interictal()
        neg_interictal = return__avg_magntiude_neg_response_interictal()
        pos_ictal = return__avg_magntiude_pos_response_ictal()
        neg_ictal = return__avg_magntiude_neg_response_ictal()

        if collect_baseline_responders:
            results.avg_baseline_responders_magnitude = {'baseline_positive': [val for i, val in enumerate(pos_baseline) if (not np.isnan(val) and not np.isnan(pos_interictal[i]))],
                                                            'baseline_negative': [val for i, val in enumerate(neg_baseline) if (not np.isnan(val) and not np.isnan(neg_interictal[i]))],
                                                            'interictal_positive': [val for i, val in enumerate(pos_interictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                                            'interictal_negative': [val for i, val in enumerate(neg_interictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))],
                                                            'ictal_positive': [val for i, val in enumerate(pos_ictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                                            'ictal_negative': [val for i, val in enumerate(neg_ictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))]
                                                         }

        else:
            results.avg_responders_magnitude = {'baseline_positive': [val for i, val in enumerate(pos_baseline) if (not np.isnan(val) and not np.isnan(pos_interictal[i]))],
                                                'baseline_negative': [val for i, val in enumerate(neg_baseline) if (not np.isnan(val) and not np.isnan(neg_interictal[i]))],
                                                'interictal_negative': [val for i, val in enumerate(neg_interictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))],
                                                'interictal_positive': [val for i, val in enumerate(pos_interictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                                'ictal_negative': [val for i, val in enumerate(neg_ictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))],
                                                'ictal_positive': [val for i, val in enumerate(pos_ictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                                }
        results.save_results()


    @staticmethod
    def plot__avg_magnitude_response(results: PhotostimResponsesNonTargetsResults, plot_baseline_responders= False):
        """plot bar plot of avg magnitude of statistically significant responders across baseline and interictal, split up by positive and negative responders
        :param results:
        :param plot_baseline_responders: if True, for post-4ap exp, use the baseline responders' avgtraces magnitude for interictal and ictal groups

        """

        if plot_baseline_responders:
            results_to_plot= results.avg_baseline_responders_magnitude
            title = ' - matched baseline responders'
        else:
            results_to_plot = results.avg_responders_magnitude
            title = '- within condition'

        pplot.plot_bar_with_points(data=[results_to_plot['baseline_positive'], results_to_plot['interictal_positive'], results_to_plot['ictal_positive']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal', 'ictal'],
                                   colors=['blue', 'green', 'purple'], y_label='Avg. magnitude of response', title='Positive responders' + title, bar=False, ylims=[-0.1, 1.0])

        pplot.plot_bar_with_points(data=[results_to_plot['baseline_negative'], results_to_plot['interictal_negative'], results_to_plot['ictal_negative']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal', 'ictal'],
                                   colors=['blue', 'green', 'purple'], y_label='Avg. magnitude of response', title='Negative responders' + title, bar=False, ylims=[-1.0, 0.1])



    # 2.3) PLOT -- BAR PLOT OF AVG TOTAL NUMBER OF POS. AND NEG RESPONSIVE CELLS
    @staticmethod
    def collect__avg_num_response(results: PhotostimResponsesNonTargetsResults):
        """
        collect: avg num of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_pos_responders(**kwargs):
            """return avg magnitude of positive responders of stim trials - pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.pre4ap_num_pos / len(expobj.s2p_nontargets_analysis)) * 100, 2)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_neg_responders(**kwargs):
            """return avg magnitude of negitive responders of stim trials - pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.pre4ap_num_neg / len(expobj.s2p_nontargets_analysis)) * 100, 2)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_pos_responders_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.post4ap_num_pos_interictal / len(expobj.s2p_nontargets_analysis)) * 100, 2)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_neg_responders_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.post4ap_num_neg_interictal / len(expobj.s2p_nontargets_analysis)) * 100, 2)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_pos_responders_ictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - ictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.post4ap_num_pos_ictal / len(expobj.s2p_nontargets_analysis)) * 100, 2)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_neg_responders_ictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - ictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return round((expobj.PhotostimResponsesNonTargets.post4ap_num_neg_ictal / len(expobj.s2p_nontargets_analysis)) * 100, 2)


        pos_baseline = return__num_pos_responders()
        neg_baseline = return__num_neg_responders()
        pos_interictal = return__num_pos_responders_interictal()
        neg_interictal = return__num_neg_responders_interictal()
        pos_ictal = return__num_pos_responders_ictal()
        neg_ictal = return__num_neg_responders_ictal()

        results.avg_responders_num = {'baseline_positive': [val for i, val in enumerate(pos_baseline) if (not np.isnan(val) and not np.isnan(pos_interictal[i]))],
                                      'baseline_negative': [val for i, val in enumerate(neg_baseline) if (not np.isnan(val) and not np.isnan(neg_interictal[i]))],
                                      'interictal_positive': [val for i, val in enumerate(pos_interictal) if
                                                              (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                      'interictal_negative': [val for i, val in enumerate(neg_interictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))],
                                      'ictal_positive': [val for i, val in enumerate(pos_ictal) if
                                                              (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                      'ictal_negative': [val for i, val in enumerate(neg_ictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))]
                                      }
        results.save_results()

    @staticmethod
    def plot__avg_num_responders(results: PhotostimResponsesNonTargetsResults):
        """plot bar plot of avg number of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        pplot.plot_bar_with_points(data=[results.avg_responders_num['baseline_positive'], results.avg_responders_num['interictal_positive'], results.avg_responders_num['ictal_positive']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal', 'ictal'],
                                   colors=['blue', 'green', 'purple'], y_label='Avg. responders (%)', title='Positive responders - within condition', bar=False, ylims=[0, 50])

        pplot.plot_bar_with_points(data=[results.avg_responders_num['baseline_negative'], results.avg_responders_num['interictal_negative'], results.avg_responders_num['ictal_negative']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal', 'ictal'],
                                   colors=['blue', 'green', 'purple'], y_label='Avg. responders (%)', title='Negative responders - within condition', bar=False, ylims=[0, 50])



    # 3) ANALYSIS OF TOTAL EVOKED RESPONSES OF NETWORK #################################################################
    def _calculate__summed_responses(self):
        """calculate total responses of significantly responding nontargets."""
        if 'pre' in self.expobj_exptype:
            __positive_responders_responses = self.adata.X[self.adata.obs['positive_responder_baseline']]
            summed_response_positive_baseline = list(np.sum(__positive_responders_responses, axis=0))  #: summed response across all positive responders at each photostim trial

            __negative_responders_responses = self.adata.X[self.adata.obs['negative_responder_baseline']]
            summed_response_negative_baseline = list(np.sum(__negative_responders_responses, axis=0))  #: summed response across all negative responders at each photostim trial

            network_summed_activity = list(np.sum(self.adata.X, axis=0))  #: summed responses across all nontargets at each photostim trial

            # add as var to anndata
            self.adata.add_variable(var_name='summed_response_pos_baseline', values=summed_response_positive_baseline)
            self.adata.add_variable(var_name='summed_response_neg_baseline', values=summed_response_negative_baseline)
            self.adata.add_variable(var_name='total_nontargets_responses', values=network_summed_activity)

        elif 'post' in self.expobj_exptype:
            network_summed_activity = list(np.sum(self.adata.X, axis=0))

            # interictal
            __positive_responders_responses = self.adata.X[self.adata.obs['positive_responder_interictal']]
            summed_response_positive_interictal = list(np.sum(__positive_responders_responses, axis=0))  #: summed response across all positive responders at each photostim trial

            __negative_responders_responses = self.adata.X[self.adata.obs['negative_responder_interictal']]
            summed_response_negative_interictal = list(np.sum(__negative_responders_responses, axis=0))  #: summed response across all negative responders at each photostim trial


            # ictal
            __positive_responders_responses = self.adata.X[self.adata.obs['positive_responder_ictal']]
            summed_response_positive_ictal = list(np.sum(__positive_responders_responses, axis=0))  #: summed response across all positive responders at each photostim trial

            __negative_responders_responses = self.adata.X[self.adata.obs['negative_responder_ictal']]
            summed_response_negative_ictal = list(np.sum(__negative_responders_responses, axis=0))  #: summed response across all negative responders at each photostim trial

            # add as var to anndata
            self.adata.add_variable(var_name='total_nontargets_responses', values=network_summed_activity)

            self.adata.add_variable(var_name='summed_response_pos_interictal', values=summed_response_positive_interictal)
            self.adata.add_variable(var_name='summed_response_neg_interictal', values=summed_response_negative_interictal)

            self.adata.add_variable(var_name='summed_response_pos_ictal', values=summed_response_positive_ictal)
            self.adata.add_variable(var_name='summed_response_neg_ictal', values=summed_response_negative_ictal)

    def _calculate__summed_responses_targets(self, expobj: Union[alloptical, Post4ap]):
        """calculate total summed dFF responses of SLM targets of experiments to compare with summed responses of nontargets."""

        if 'pre' in self.expobj_exptype:
            summed_responses = list(np.sum(expobj.PhotostimResponsesSLMTargets.adata.X, axis=0))

            expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='summed_response_SLMtargets', values=summed_responses)

            # return summed_responses

        elif 'post' in self.expobj_exptype:
            summed_responses = list(np.sum(expobj.PhotostimResponsesSLMTargets.adata.X, axis=0))

            expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='summed_response_SLMtargets', values=summed_responses)

            # return summed_responses

    @staticmethod
    def run__summed_responses(rerun=0):
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=rerun, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def _run__summed_responses(**kwargs):
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            expobj.PhotostimResponsesNonTargets._calculate__summed_responses()
            expobj.PhotostimResponsesNonTargets._calculate__summed_responses_targets(expobj=expobj)
            expobj.save()

        _run__summed_responses()

    # 3.1) plot - scatter plot of total evoked activity on trial vs. total activity of SLM targets on same trial - split up based on groups
    @staticmethod
    def plot__summed_activity_vs_targets_activity(results: PhotostimResponsesNonTargetsResults):

        # pre4ap - baseline
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, allow_rerun=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def collect_summed_responses_baseline(**kwargs):
            expobj = kwargs['expobj']

            summed_responses = pd.DataFrame({'exp': [expobj.t_series_name] * expobj.PhotostimResponsesSLMTargets.adata.n_vars,
                                             'targets': expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets'],
                                             'non-targets_pos': expobj.PhotostimResponsesNonTargets.adata.var['summed_response_pos_baseline'],
                                             'non-targets_neg': expobj.PhotostimResponsesNonTargets.adata.var['summed_response_neg_baseline'],
                                             'all_non-targets': expobj.PhotostimResponsesNonTargets.adata.var['total_nontargets_responses']})

            # z scoring of all collected responses
            network_summed_activity_zsco = np.round(stats.zscore(summed_responses['all_non-targets'], ddof=1), 3)
            targets_summed_activity_zsco = np.round(stats.zscore(summed_responses['targets'], ddof=1), 3)

            # calculating linear regression metrics between summed targets and summed total network for each experiment
            slope, intercept, r_value, p_value, std_err = stats.linregress(x=targets_summed_activity_zsco,
                                                                           y=network_summed_activity_zsco)
            regression_y = slope * targets_summed_activity_zsco + intercept

            summed_responses['targets_summed_zscored'] = targets_summed_activity_zsco
            summed_responses['all_non-targets_zscored'] = network_summed_activity_zsco
            summed_responses['all_non-targets_score_regression'] = regression_y


            lin_reg_scores = pd.DataFrame({
                'exp': expobj.t_series_name,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'mean_targets': np.mean(summed_responses['targets']),
                'mean_non-targets': np.mean(summed_responses['all_non-targets']),
                'std_targets': np.std(summed_responses['targets'], ddof=1),
                'std_non-targets': np.std(summed_responses['all_non-targets'], ddof=1)
            }, index=[expobj.t_series_name])


            return summed_responses, lin_reg_scores
            # return expobj.PhotostimResponsesNonTargets.adata.var['summed_response_pos_baseline'], expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets']

        func_collector_baseline = collect_summed_responses_baseline()

        if func_collector_baseline is not None:
            summed_responses_baseline = pd.DataFrame({'exp': [], 'targets': [], 'non-targets_pos': [], 'non-targets_neg': [], 'all_non-targets': [],
                                                      'targets_summed_zscored': [], 'all_non-targets_zscored': [], 'all_non-targets_score_regression': []})
            lin_reg_scores_baseline = pd.DataFrame({'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})

            for exp in func_collector_baseline:
                summed_responses_baseline = pd.concat([summed_responses_baseline, exp[0]])
                lin_reg_scores_baseline = pd.concat([lin_reg_scores_baseline, exp[1]])

            summed_responses_baseline.shape

            results.summed_responses = {'baseline': summed_responses_baseline,
                                 }

            results.lin_reg_summed_responses = {'baseline': lin_reg_scores_baseline,
                                 }

            results.save_results()

        # post4ap - interictal
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def collect_summed_responses_interictal(lin_reg_scores, **kwargs):
            expobj = kwargs['expobj']

            summed_responses = pd.DataFrame({'exp': [expobj.t_series_name] * sum([expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'interictal'][0]),
                                             'targets': expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'interictal'],
                                             'non-targets_pos': expobj.PhotostimResponsesNonTargets.adata.var['summed_response_pos_interictal'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'interictal'],
                                             'non-targets_neg': expobj.PhotostimResponsesNonTargets.adata.var['summed_response_neg_interictal'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'interictal'],
                                             'all_non-targets': expobj.PhotostimResponsesNonTargets.adata.var['total_nontargets_responses'][expobj.PhotostimResponsesSLMTargets.adata.var['stim_group'] == 'interictal']})


            # z scoring of all collected responses
            network_summed_activity_zsco = np.round(stats.zscore(summed_responses['all_non-targets'], ddof=1), 3)
            targets_summed_activity_zsco = np.round(stats.zscore(summed_responses['targets'], ddof=1), 3)

            # z scoring to mean and std of baseline group of same experiment
            from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze
            for map_key, expid in AllOpticalExpsToAnalyze.trial_maps['post'].items():  # find the pre4ap exp that matches with the current post4ap experiment
                if expobj.t_series_name in expid:
                    pre4ap_match_id = AllOpticalExpsToAnalyze.trial_maps['pre'][map_key]
                    if map_key == 'g':
                        pre4ap_match_id = AllOpticalExpsToAnalyze.trial_maps['pre'][map_key][1]
                    break
            network_summed_activity_zsco = np.round([( x - float(lin_reg_scores.loc[pre4ap_match_id, 'mean_non-targets'])) / float(lin_reg_scores.loc[pre4ap_match_id, 'std_non-targets']) for x in list(summed_responses['all_non-targets'])], 3)
            targets_summed_activity_zsco = np.round([( x - float(lin_reg_scores.loc[pre4ap_match_id, 'mean_targets'])) / float(lin_reg_scores.loc[pre4ap_match_id, 'std_targets']) for x in list(summed_responses['targets'])], 3)

            # ------ exclude datapoints that are >5 z score points (or < -5):
            include_idx = [idx for idx, zscore in enumerate(targets_summed_activity_zsco) if -5 < zscore < 5]
            if len(include_idx) < len(targets_summed_activity_zsco):
                print(f'**** excluding {len(targets_summed_activity_zsco) - len(include_idx)} stims from exp: {expobj.t_series_name} ****')
            targets_summed_activity_zsco = np.array([targets_summed_activity_zsco[i] for i in include_idx])
            network_summed_activity_zsco = np.array([network_summed_activity_zsco[i] for i in include_idx])

            # calculating linear regression metrics between summed targets and summed total network for each experiment
            slope, intercept, r_value, p_value, std_err = stats.linregress(x=targets_summed_activity_zsco,
                                                                           y=network_summed_activity_zsco)
            regression_y = slope * targets_summed_activity_zsco + intercept

            summed_responses_zscore = pd.DataFrame({'exp': [expobj.t_series_name] * len(regression_y),
                                                    'targets_summed_zscored': targets_summed_activity_zsco,
                                                    'all_non-targets_zscored': network_summed_activity_zsco,
                                                    'all_non-targets_score_regression': regression_y})

            # summed_responses['targets_summed_zscored'] = targets_summed_activity_zsco
            # summed_responses['all_non-targets_zscored'] = network_summed_activity_zsco
            # summed_responses['all_non-targets_score_regression'] = regression_y


            lin_reg_scores = pd.DataFrame({
                'exp': expobj.t_series_name,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value
            }, index=[expobj.t_series_name])



            return summed_responses_zscore, lin_reg_scores
            # return expobj.PhotostimResponsesNonTargets.adata.var['summed_response_pos_interictal'], expobj.PhotostimResponsesSLMTargets.adata.var['summed_response_SLMtargets']

        func_collector_interictal = collect_summed_responses_interictal(lin_reg_scores=results.lin_reg_summed_responses['baseline'])

        # summed_responses_interictal = pd.DataFrame({'exp': [], 'targets': [], 'non-targets_pos': [], 'non-targets_neg': [], 'all_non-targets': [],
        #                                           'targets_summed_zscored': [], 'all_non-targets_zscored': [], 'all_non-targets_score_regression': []})

        summed_responses_interictal_zscore = pd.DataFrame({'exp': [], 'targets_summed_zscored': [], 'all_non-targets_zscored': [], 'all_non-targets_score_regression': []})
        lin_reg_scores_interictal = pd.DataFrame({'exp': [], 'slope': [], 'intercept': [], 'r_value': [], 'p_value': []})

        for exp in func_collector_interictal:
            summed_responses_interictal_zscore = pd.concat([summed_responses_interictal_zscore, exp[0]])
            lin_reg_scores_interictal = pd.concat([lin_reg_scores_interictal, exp[1]])

        results.lin_reg_summed_responses['interictal'] = lin_reg_scores_interictal
        results.summed_responses['interictal'] = summed_responses_interictal_zscore
        results.save_results()

        summed_responses_interictal_zscore.shape



        # make plots

        # SCATTER PLOT OF DATAPOINTS
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=results.summed_responses['baseline']['targets_summed_zscored'],
                                                                       y=results.summed_responses['baseline']['all_non-targets_zscored'])
        regression_y = slope * results.summed_responses['baseline']['targets_summed_zscored'] + intercept

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
        fig, axs[0] = pplot.make_general_scatter(x_list=[results.summed_responses['baseline']['targets_summed_zscored']],
                                                 y_data=[results.summed_responses['baseline']['all_non-targets_zscored']], figsize=(6.5, 4), fig=fig, ax=axs[0],
                                                 s=50,facecolors=['gray'], edgecolors=['black'], lw=1, alpha=1,
                                                 x_labels=['total targets activity'], y_labels=['total network activity'],
                                                 legend_labels=[f'baseline - $R^2$: {r_value ** 2:.2e}, p = {p_value**2:.2e}'], show=False)

        axs[0].plot(results.summed_responses['baseline']['targets_summed_zscored'], regression_y, color='black')


        slope, intercept, r_value, p_value, std_err = stats.linregress(x=summed_responses_interictal_zscore['targets_summed_zscored'],
                                                                       y=summed_responses_interictal_zscore['all_non-targets_zscored'])

        regression_y = slope * summed_responses_interictal_zscore['targets_summed_zscored'] + intercept

        pplot.make_general_scatter(x_list = [summed_responses_interictal_zscore['targets_summed_zscored']],
                                   y_data=[summed_responses_interictal_zscore['all_non-targets_zscored']], s=50, facecolors=['green'],
                                   edgecolors=['black'], lw=1, alpha=1, x_labels=['total targets activity'],
                                   y_labels=['total network activity'], fig = fig, ax= axs[1], legend_labels=[f'interictal - $R^2$: {r_value**2:.2e}, p = {p_value**2:.2e}'], show = False)

        axs[1].plot(summed_responses_interictal_zscore['targets_summed_zscored'], regression_y, color = 'black')

        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_ylim([-15, 15])
        axs[1].set_ylim([-15, 15])
        axs[0].set_xlim([-7, 7])
        axs[1].set_xlim([-7, 7])
        fig.suptitle('Total z-scored (to baseline) responses for all trials, all exps', wrap = True)
        fig.tight_layout(pad=0.6)
        fig.show()


        # BAR PLOT OF PEARSON'S R CORR VALUES BETWEEN BASELINE AND INTERICTAL
        pplot.plot_bar_with_points(data=[[i**2 for i in results.lin_reg_summed_responses['baseline']['r_value']],
                                         [i**2 for i in lin_reg_scores_interictal['r_value']]],
                                   paired = True, bar = False, colors=['black', 'green'], edgecolor='black', lw=1,
                                   x_tick_labels=['Baseline', 'Interictal'], ylims=[0, 1], y_label='$R^2$', title='$R^2$ value per experiment')



if __name__ == '__main__':
    # expobj: alloptical = Utils.import_expobj(exp_prep='RL108 t-009')
    # expobj: Post4ap = Utils.import_expobj(exp_prep='RL108 t-013')

    main = PhotostimResponsesAnalysisNonTargets
    results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

    # main.run__initPhotostimResponsesAnalysisNonTargets()

    # main.run__plot_sig_responders_traces(plot_baseline_responders=True)
    # main.run__plot_sig_responders_traces(plot_baseline_responders=False)
    # main.run__create_anndata()
    # main.run__classify_and_measure_nontargets_szboundary(force_redo=False)


    # 2) basic plotting of responders baseline, interictal, and ictal
    # main.collect__avg_magnitude_response(results=results, collect_baseline_responders=True)
    # results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()
    # main.plot__avg_magnitude_response(results=results, plot_baseline_responders=True)

    # main.collect__avg_magnitude_response(results=results, collect_baseline_responders=False)
    # results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()
    main.plot__avg_magnitude_response(results=results, plot_baseline_responders=False)  # < - there is one experiment that doesn't have any responders i think....


    # main.collect__avg_num_response(results=results)
    main.plot__avg_num_responders(results=results)


    #
    #
    #3) calculate summed responses and plot against evoked targets' activity
    # main.run__summed_responses(rerun=0)
    # main.plot__summed_activity_vs_targets_activity(results=results)


