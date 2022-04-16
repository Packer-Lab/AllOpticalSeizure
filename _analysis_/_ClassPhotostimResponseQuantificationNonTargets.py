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
        self.avg_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal

REMAKE = False
if not os.path.exists(PhotostimResponsesNonTargetsResults.SAVE_PATH) or REMAKE:
    results = PhotostimResponsesNonTargetsResults()
    results.save_results()




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

######

class PhotostimResponsesQuantificationNonTargets(Quantification):
    """class for quanitying responses of non-targeted cells at photostimulation trials.
    non-targeted cells classified as Suite2p ROIs that were not SLM targets.


    Tasks:
    [ ] scatter plot of individual stim trials: response magnitude of targets vs. response magnitude of all (significantly responding) nontargets
                                                                                - (maybe, evoked summed magnitude of pos and neg sig. responders - check previous plots on this to see if there's some insights there....)
        [x] set up code to split post4ap nontargets responses collected data into interictal and ictal
        [x] continue working thorugh workflow to collect traces for pos and neg sig. responders - pre4ap and post4ap
        [x] need to think of a way to normalize results within experiments before collating across experiments
            [x] maybe calcualte Pearson's r for each experiment between baseline and interictal, and compare these values across experiments
            [x] or simple z - scoring of targets and non-targets responses --> think I'll try this first - do for one experiment and plot results
        [x] also might consider measuring activity responses of ALL cells, not just the significantly responding nontargets < -- seems to be a promising avenue

        [ ] plan: calculate $R^2$ for each experiment (pre4ap and post4ap interictal), and compare on bar plot
            [ ] - could also consider aggregating z scores across all experiments for baseline and interictal stims

    [ ] code for splitting nontargets inside and outside sz boundary

    """

    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    mean_photostim_responses_baseline: List[float] = None
    mean_photostim_responses_interictal: List[float] = None
    mean_photostim_responses_ictal: List[float] = None
    EXCLUDE_TRIALS = ['PS04 t-012',  # no responding cells and also doubled up by PS04 t-017 in any case
                      ]

    def __init__(self, expobj: Union[alloptical, Post4ap]):

        self.diff_responses = None  #: subtraction of stim responses post array with stim responses pre array (direct diff. of post-stim - pre-stim value for each nontarget across all stims)


        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimResponsesNonTargets MODULE to expobj: {expobj.t_series_name}')
        self._allopticalAnalysisNontargets(expobj=expobj, normalize_to='pre-stim', to_plot=False)
        if not hasattr(expobj, 's2p_nontargets'):
            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(force_redo=True, plot=False)
            expobj.save()


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0, skip_trials=EXCLUDE_TRIALS)
    def run__initPhotostimResponsesQuantificationNonTargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets = PhotostimResponsesQuantificationNonTargets(expobj=expobj)
        expobj.save()

    def __repr__(self):
        return f"PhotostimResponsesNonTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0) ANALYSIS OF NON-TARGETS IN ALL OPTICAL EXPERIMENTS.

    def _allopticalAnalysisNontargets(self, expobj: Union[alloptical, Post4ap], normalize_to='pre_stim', to_plot=True, save_plot_suffix=''):
        if 'pre' in expobj.exptype:
            self.diff_responses, self.wilcoxons, self.sig_units = expobj._trialProcessing_nontargets(normalize_to='pre-stim', stims = 'all', fdr_alpha=0.25)
            # self.sig_units = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1, save=False)  #: array of bool describing statistical significance of responder

            self.dff_stimtraces = expobj.dff_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dff_avgtraces = expobj.dff_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet
            self.dfstdF_stimtraces = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dfstdF_avgtraces = expobj.dfstdF_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet

        elif 'post' in expobj.exptype:
            # all stims
            self.diff_responses, self.wilcoxons, self.sig_units = expobj._trialProcessing_nontargets(normalize_to='pre-stim',
                                                                                                     stims='all', fdr_alpha=0.25)
            self.dff_stimtraces = expobj.dff_traces_nontargets

            # interictal stims only
            self.diff_responses_interictal, self.wilcoxons_interictal, self.sig_units_interictal = expobj._trialProcessing_nontargets(normalize_to='pre-stim',
                                                                                                                                      stims=expobj.stims_out_sz, fdr_alpha=0.25)
            # self.sig_units_interictal = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1, save=False)
            self.dff_stimtraces_interictal = expobj.dff_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dff_avgtraces_interictal = expobj.dff_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet
            self.dfstdF_stimtraces_interictal = expobj.dfstdF_traces_nontargets  #: all stim timed trace snippets for all nontargets, shape: # cells x # stims x # frames of trace snippet
            self.dfstdF_avgtraces_interictal = expobj.dfstdF_traces_nontargets_avg  #: avg of trace snippets from all stims for all nontargets, shape: # cells x # frames of trace snippet



            # ictal stims only
            self.diff_responses_ictal, self.wilcoxons_ictal, self.sig_units_ictal = expobj._trialProcessing_nontargets(normalize_to='pre-stim',
                                                                                                                       stims=expobj.stims_in_sz, fdr_alpha=0.25)
            # self.sig_units_ictal = expobj._sigTestAvgResponse_nontargets(p_vals=self.wilcoxons, alpha=0.1)


        # # make figure containing plots showing average responses of nontargets to photostim
        # # save_plot_path = expobj.analysis_save_path[:30] + 'Results_figs/' + save_plot_suffix
        # fig_non_targets_responses(expobj=expobj, plot_subset=False,
        #                           save_fig_suffix=save_plot_suffix) if to_plot else None

        print('\n** FIN. * allopticalAnalysisNontargets * %s %s **** ' % (
            expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        print(
            '-------------------------------------------------------------------------------------------------------------\n\n')

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0, skip_trials=EXCLUDE_TRIALS)
    def run_allopticalNontargets(**kwargs):
        expobj = kwargs['expobj']
        # if not hasattr(expobj, 's2p_nontargets'):
        #     expobj._parseNAPARMgpl()
        #     expobj._findTargetsAreas()
        #     expobj._findTargetedS2pROIs(force_redo=True, plot=False)
        #     expobj.save()

        # expobj.PhotostimAnalysisSlmTargets.pre_stim_fr = int(expobj.PhotostimAnalysisSlmTargets._pre_stim_sec * expobj.fps)  # length of pre stim trace collected (in frames)
        # expobj.PhotostimAnalysisSlmTargets.post_stim_fr = int(expobj.PhotostimAnalysisSlmTargets._post_stim_sec * expobj.fps)  # length of post stim trace collected (in frames)
        # expobj.PhotostimAnalysisSlmTargets.pre_stim_response_frames_window = int(
        #     expobj.fps * expobj.PhotostimAnalysisSlmTargets.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)
        # expobj.PhotostimAnalysisSlmTargets.post_stim_response_frames_window = int(
        #     expobj.fps * expobj.PhotostimAnalysisSlmTargets.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)



        # expobj.PhotostimResponsesNonTargets._allopticalAnalysisNontargets(expobj=expobj, normalize_to='pre-stim', to_plot=False)
        # expobj.save()


        #### running other functions:

        # collect traces of statistically significant followers:
        expobj.PhotostimResponsesNonTargets.collect__sig_responders_responses(expobj=expobj)

        # plot traces of statistically significant followers:
        # expobj.PhotostimResponsesNonTargets.plot__sig_responders_traces(expobj=expobj)


        # anndata
        # expobj.PhotostimResponsesNonTargets.create_anndata(expobj=expobj)

        expobj.save()

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
                     'aspect_ratio', 'npix_norm', 'skew', 'std'], index=range((len(expobj.s2p_nontargets)) - len(expobj.s2p_nontargets_exclude)))
        for i, idx in enumerate(obs_meta.index):
            if idx not in expobj.s2p_nontargets_exclude:
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
            for i, idx in enumerate(expobj.s2p_nontargets):
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
    def run__create_anndata(rerun=0):
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

    # 2) COLLECT pos/neg sig. responders traces and responses

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

    def collect__sig_responders_responses(self, expobj: Union[alloptical, Post4ap]):
        """
        Collect responses traces of statistically significant positive and negative photostimulation timed followers. Also collect response magnitude of all pos. and neg. responders for all stims.


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
            pre_stim_fr = expobj.PhotostimAnalysisSlmTargets.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = expobj.PhotostimAnalysisSlmTargets.post_stim_fr  # setting the post_stim array collection period again hard



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
            # post4ap_possig_responders_avgtrace = []
            # post4ap_negsig_responders_avgtrace = []
            # post4ap_num_pos = 0
            # post4ap_num_neg = 0


            self.post4ap_possig_responders_responses_interictal = self.diff_responses_interictal[self.sig_units_interictal][self.pos_sig_responders_interictal]  #: response magnitude for all pos responders for all stims
            self.post4ap_negsig_responders_responses_interictal = self.diff_responses_interictal[self.sig_units_interictal][self.neg_sig_responders_interictal]  #: response magnitude for all neg responders for all stims


            self.post4ap_possig_responders_traces = self.dfstdF_stimtraces_interictal[self.sig_units_interictal][np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) > 0)[0]]
            self.post4ap_negsig_responders_traces = self.dfstdF_stimtraces_interictal[self.sig_units_interictal][np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) < 0)[0]]


            ## MAKE ARRAY OF TRACE SNIPPETS THAT HAVE PHOTOSTIM PERIOD ZERO'D

            # post4ap_possig_responders_avgtrace.append(post4ap_possig_responders_avgtrace_)
            # post4ap_negsig_responders_avgtrace.append(post4ap_negsig_responders_avgtrace__)
            stim_dur_fr = int(np.ceil(0.250 * expobj.fps))  # setting 250ms as the dummy standardized stimduration
            pre_stim_fr = expobj.PhotostimAnalysisSlmTargets.pre_stim_fr  # setting the pre_stim array collection period
            post_stim_fr = expobj.PhotostimAnalysisSlmTargets.post_stim_fr  # setting the post_stim array collection period again hard



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
            # print('stop here... [5.5-1]')


            # negative responders
            post4ap_negsig_responders_avgtraces = np.mean(self.post4ap_negsig_responders_traces, axis=1)
            data_traces = []
            for trace in post4ap_negsig_responders_avgtraces:
                trace_ = trace[expobj.pre_stim - pre_stim_fr: expobj.pre_stim]
                trace_ = np.append(trace_, [[0] * stim_dur_fr])
                trace_ = np.append(trace_, trace[
                                           expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                data_traces.append(trace_)
            self.post4ap_negsig_responders_avgtraces_interictal = np.array(data_traces);
            print(f"shape of post4ap_negsig_responders trace array {self.post4ap_negsig_responders_avgtraces_interictal.shape} [2.2-2]")
            # print('stop here... [5.5-2]')


            self.post4ap_num_pos_interictal = self.post4ap_possig_responders_avgtraces_interictal.shape[0]
            self.post4ap_num_neg_interictal = self.post4ap_negsig_responders_avgtraces_interictal.shape[0]



            # return post4ap_possig_responders_avgtraces_interictal, post4ap_negsig_responders_avgtraces_interictal

            # post4ap__sig_responders_traces()

        expobj.save()


    # 2.0.1) CREATE LIST OF EXPERIMENTS WITH RESPONDERS TO ANALYZE
    @staticmethod
    def collect__exp_responders(results: PhotostimResponsesNonTargetsResults):
        """collect experiments that will be used for non targets analysis. these experiments have to have atleast """

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

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, set_cache=0, skip_trials=EXCLUDE_TRIALS, run_trials=['PS07 t-007', 'PS07 t-011'])
    def run__plot_sig_responders_traces(**kwargs):
        expobj = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets.plot__sig_responders_traces(expobj=expobj)


    # 2.2) PLOT -- BAR PLOT OF AVG MAGNITUDE OF RESPONSE

    @staticmethod
    def collect__avg_magnitude_response(results: PhotostimResponsesNonTargetsResults):
        """plot bar plot of avg magnitude of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_pos_response(**kwargs):
            """return avg magnitude of positive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_possig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_pos_response_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_interictal)

        # @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        # def return__avg_magntiude_pos_response(**kwargs):
        #     """return avg magnitude of positive responders of stim trials - ictal"""
        #     expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        #     return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_ictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_neg_response(**kwargs):
            """return avg magnitude of negitive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_negsig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__avg_magntiude_neg_response_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_responses_interictal)

        pos_baseline = return__avg_magntiude_pos_response()
        pos_interictal = return__avg_magntiude_pos_response_interictal()
        neg_baseline = return__avg_magntiude_neg_response()
        neg_interictal = return__avg_magntiude_neg_response_interictal()

        results.avg_responders_magnitude = {'baseline_positive': [val for i, val in enumerate(pos_baseline) if (not np.isnan(val) and not np.isnan(pos_interictal[i]))],
                                            'interictal_positive': [val for i, val in enumerate(pos_interictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                            'baseline_negative': [val for i, val in enumerate(neg_baseline) if (not np.isnan(val) and not np.isnan(neg_interictal[i]))],
                                            'interictal_negative': [val for i, val in enumerate(neg_interictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))]}
        results.save_results()


    @staticmethod
    def plot__avg_magnitude_response(results: PhotostimResponsesNonTargetsResults):
        """plot bar plot of avg magnitude of statistically significant responders across baseline and interictal, split up by positive and negative responders"""

        pplot.plot_bar_with_points(data=[results.avg_responders_magnitude['baseline_positive'], results.avg_responders_magnitude['interictal_positive']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['blue', 'green'], y_label='Avg. magnitude of response', title='Positive responders', bar=False, ylims=[0, 1.0])

        pplot.plot_bar_with_points(data=[results.avg_responders_magnitude['baseline_negative'], results.avg_responders_magnitude['interictal_negative']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['blue', 'green'], y_label='Avg. magnitude of response', title='Negative responders', bar=False, ylims=[-1.0, 0])


    # 2.3) PLOT -- BAR PLOT OF AVG TOTAL NUMBER OF POS. AND NEG RESPONSIVE CELLS

    @staticmethod
    def collect__avg_num_response(results: PhotostimResponsesNonTargetsResults):
        """collect: avg num of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_pos_responders(**kwargs):
            """return avg magnitude of positive responders of stim trials - pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.pre4ap_num_pos

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_pos_responders_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.post4ap_num_pos_interictal

        # @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        # def return__avg_magntiude_pos_response(**kwargs):
        #     """return avg magnitude of positive responders of stim trials - ictal"""
        #     expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        #     return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_ictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_neg_responders(**kwargs):
            """return avg magnitude of negitive responders of stim trials - pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.pre4ap_num_neg

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0, skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS)
        def return__num_neg_responders_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.post4ap_num_neg_interictal

        pos_baseline = return__num_pos_responders()
        pos_interictal = return__num_pos_responders_interictal()
        neg_baseline = return__num_neg_responders()
        neg_interictal = return__num_neg_responders_interictal()

        results.avg_responders_num = {'baseline_positive': [val for i, val in enumerate(pos_baseline) if (not np.isnan(val) and not np.isnan(pos_interictal[i]))],
                                      'interictal_positive': [val for i, val in enumerate(pos_interictal) if (not np.isnan(val) and not np.isnan(pos_baseline[i]))],
                                      'baseline_negative': [val for i, val in enumerate(neg_baseline) if (not np.isnan(val) and not np.isnan(neg_interictal[i]))],
                                      'interictal_negative': [val for i, val in enumerate(neg_interictal) if (not np.isnan(val) and not np.isnan(neg_baseline[i]))]}
        results.save_results()



    @staticmethod
    def plot__avg_num_responders(results: PhotostimResponsesNonTargetsResults):
        """plot bar plot of avg number of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        pplot.plot_bar_with_points(data=[results.avg_responders_num['baseline_positive'], results.avg_responders_num['interictal_positive']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['blue', 'green'], y_label='Avg. number of responders', title='Positive responders', bar=False, ylims=[0, 600])

        pplot.plot_bar_with_points(data=[results.avg_responders_num['baseline_negative'], results.avg_responders_num['interictal_negative']],
                                   paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['blue', 'green'], y_label='Avg. number of responders', title='Negative responders', bar=False, ylims=[0, 600])


    # 3)
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
        """calculate total summed responses of SLM targets of experiments to compare with summed responses of nontargets."""

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




# %%
if __name__ == '__main__':
    # expobj = Utils.import_expobj(exp_prep='RL108 t-009')

    main = PhotostimResponsesQuantificationNonTargets
    results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()


    # main.run__initPhotostimResponsesQuantificationNonTargets()
    # main.run_allopticalNontargets()  # <-- using as pipeline right now to test multiple functions for one exp
    # # main.run__plot_sig_responders_traces()
    # # main.run__create_anndata()
    # # main.run__plot_sig_responders_traces()
    #
    # # 2) basic plotting of responders pre4ap and interictal
    # # main.collect__avg_magnitude_response(results=results)
    # # main.plot__avg_magnitude_response(results=results)
    # # main.collect__avg_num_response(results=results)
    # # main.plot__avg_num_responders(results=results)
    #
    #
    # #3) calculate summed responses and plot against evoked targets' activity
    main.run__summed_responses(rerun=0)
    main.plot__summed_activity_vs_targets_activity(results=results)


    # main.run__fix_anndata()


    # %%
    # results -- calculate pos/neg sig. responders avg response


    # %%

    # expobj = Utils.import_expobj(exp_prep='RL108 t-009')





# %% ARCHIVE

# def plot_dff_significant_pos_neg_responders(self, expobj: Union[alloptical, Post4ap]):
#     """Plot dFF stim traces of positive and negative significant responders.
#         - original code"""
#     import _utils_.alloptical_plotting as aoplot
#
#     print('\n----------------------------------------------------------------')
#     print('plotting dFF for significant cells ')
#     print('----------------------------------------------------------------')
#
#     expobj.sig_cells = [expobj.s2p_nontargets[i] for i, x in enumerate(expobj.sig_units) if x]
#     expobj.pos_sig_cells = [expobj.sig_cells[i] for i in
#                             np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
#     expobj.neg_sig_cells = [expobj.sig_cells[i] for i in
#                             np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
#
#     f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
#     # plot peristim avg dFF of pos_sig_cells
#     selection = [expobj.s2p_nontargets.index(i) for i in expobj.pos_sig_cells]
#     x = expobj.dff_traces_nontargets_avg[selection]
#     y_label = 'pct. dFF (normalized to prestim period)'
#     f, ax[0, 0] = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title='positive sig. responders', y_label=y_label, fig=f,
#                                                 ax=ax[0, 0], show=False,
#                                                 x_label=None, y_lims=[-50, 200])
#
#     # plot peristim avg dFF of neg_sig_cells
#     selection = [expobj.s2p_nontargets.index(i) for i in expobj.neg_sig_cells]
#     x = expobj.dff_traces_nontargets_avg[selection]
#     y_label = None
#     f, ax[0, 1] = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title='negative sig. responders', y_label=None, fig=f, ax=ax[0, 1],
#                                                 show=False,
#                                                 x_label=None, y_lims=[-50, 200])
#
#     # plot peristim avg dFstdF of pos_sig_cells
#     selection = [expobj.s2p_nontargets.index(i) for i in expobj.pos_sig_cells]
#     x = expobj.dfstdF_traces_nontargets_avg[selection]
#     y_label = 'dF/stdF'
#     f, ax[1, 0] = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title=None, y_label=y_label, fig=f, ax=ax[1, 0], show=False,
#                                                 x_label='Time (seconds) ', y_lims=[-1, 1])
#
#     # plot peristim avg dFstdF of neg_sig_cells
#     selection = [expobj.s2p_nontargets.index(i) for i in expobj.neg_sig_cells]
#     x = expobj.dfstdF_traces_nontargets_avg[selection]
#     y_label = None
#     f, ax[1, 1] = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title=None, y_label=y_label, fig=f, ax=ax[1, 1], show=False,
#                                                 x_label='Time (seconds) ', y_lims=[-1, 1])
#     f.suptitle(
#         f"{expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - significant +ve and -ve responders")
#     f.tight_layout(pad=1.3)
#     f.show()
#
# def plot_dff_significant_pos_neg_responders2(self, expobj: Union[alloptical, Post4ap]):
#     """Plot dFF stim traces of positive and negative significant responders. - version 2?
#         - original code"""
#
#     import _utils_.alloptical_plotting as aoplot
#
#     print('\n----------------------------------------------------------------')
#     print('plotting dFF for significant cells ')
#     print('----------------------------------------------------------------')
#
#     expobj.sig_cells = [expobj.s2p_nontargets[i] for i, x in enumerate(self.sig_units) if x]
#     expobj.pos_sig_cells = [expobj.sig_cells[i] for i in
#                             np.where(np.nanmean(expobj.post_array_responses[self.sig_units, :], axis=1) > 0)[0]]
#     expobj.neg_sig_cells = [expobj.sig_cells[i] for i in
#                             np.where(np.nanmean(expobj.post_array_responses[self.sig_units, :], axis=1) < 0)[0]]
#
#     f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
#     # plot peristim avg dFF of pos_sig_cells
#     # selection = [expobj.s2p_nontargets.index(i) for i in self.pos_sig_responders]
#     # dff_traces = self.dff_avgtraces[selection]
#     dff_traces = self.dff_avgtraces[self.pos_sig_responders]
#     y_label = 'pct. dFF (normalized to prestim period)'
#     f, ax[0, 0] = aoplot.plot_periphotostim_avg(arr=dff_traces, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title='positive sig. responders', y_label=y_label, fig=f,
#                                                 ax=ax[0, 0], show=False,
#                                                 x_label=None, y_lims=[-50, 200])
#
#     # plot peristim avg dFF of neg_sig_cells
#     # selection = [expobj.s2p_nontargets.index(i) for i in self.neg_sig_responders]
#     # dff_traces = self.dff_avgtraces[selection]
#     dff_traces = self.dff_avgtraces[self.neg_sig_responders]
#     y_label = None
#     f, ax[0, 1] = aoplot.plot_periphotostim_avg(arr=dff_traces, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title='negative sig. responders', y_label=None, fig=f, ax=ax[0, 1],
#                                                 show=False,
#                                                 x_label=None, y_lims=[-50, 200])
#
#     # plot peristim avg dFstdF of pos_sig_cells
#     # selection = [expobj.s2p_nontargets.index(i) for i in expobj.pos_sig_cells]
#     # dff_traces = expobj.dfstdF_traces_nontargets_avg[selection]
#     dfstdf_traces = expobj.dfstdF_traces_nontargets_avg[self.pos_sig_responders]
#     y_label = 'dF/stdF'
#     f, ax[1, 0] = aoplot.plot_periphotostim_avg(arr=dfstdf_traces, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title=None, y_label=y_label, fig=f, ax=ax[1, 0], show=False,
#                                                 x_label='Time (seconds) ', y_lims=[-1, 1])
#
#     # plot peristim avg dFstdF of neg_sig_cells
#     # selection = [expobj.s2p_nontargets.index(i) for i in expobj.neg_sig_cells]
#     # dff_traces = expobj.dfstdF_traces_nontargets_avg[selection]
#     dfstdf_traces = expobj.dfstdF_traces_nontargets_avg[self.neg_sig_responders]
#     y_label = None
#     f, ax[1, 1] = aoplot.plot_periphotostim_avg(arr=dfstdf_traces, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
#                                                 title=None, y_label=y_label, fig=f, ax=ax[1, 1], show=False,
#                                                 x_label='Time (seconds) ', y_lims=[-1, 1])
#     f.suptitle(
#         f"{expobj.metainfo['exptype']} {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - significant +ve and -ve responders")
#     f.tight_layout(pad=1.3)
#     f.show()
