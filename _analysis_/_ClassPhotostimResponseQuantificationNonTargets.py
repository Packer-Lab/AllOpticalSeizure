import os
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

# %% function definitions from alloptical_utils_pplot file - 04/05/22
###### NON TARGETS analysis + plottings


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
    [ ] scatter plot of individual stim trials: response magnitude of targets vs. response magnitude of all nontargets
                                                                                - (maybe, response magnitude of sig. responders only?)
                                                                                - (maybe, evoked summed magnitude of sig. responders only? - check previous plots on this to see if there's some insights there....)
        [x] set up code to split post4ap nontargets responses collected data into interictal and ictal
        [ ] continue working thorugh workflow to collect traces for pos and neg sig. responders - pre4ap and post4ap

    """

    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    mean_photostim_responses_baseline: List[float] = None
    mean_photostim_responses_interictal: List[float] = None
    mean_photostim_responses_ictal: List[float] = None

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
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
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

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=1, run_trials=['PS04 t-017', 'PS07 t-007'])
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



        expobj.PhotostimResponsesNonTargets._allopticalAnalysisNontargets(expobj=expobj, normalize_to='pre-stim', to_plot=False)
        # expobj.save()


        #### running other functions:

        # collect traces of statistically significant followers:
        expobj.PhotostimResponsesNonTargets.collect__sig_responders_responses(expobj=expobj)

        # plot traces of statistically significant followers:
        expobj.PhotostimResponsesNonTargets.plot__sig_responders_traces(expobj=expobj)


        # anndata
        expobj.PhotostimResponsesNonTargets.create_anndata(expobj=expobj)

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
                var_meta.loc['im_group', fr_idx] = 'ictal' if fr_idx in expobj.stims_in_sz else 'interictal'
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

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
    def run__create_anndata(**kwargs):
        expobj: Union[alloptical, Post4ap]  = kwargs['expobj']
        expobj.PhotostimResponsesNonTargets.create_anndata(expobj=expobj)
        expobj.save()


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
        return np.where(np.nanmean(self.diff_responses_interictal[self.sig_units_interictal, :], axis=1) > 0)[0]

    @property
    def pos_sig_responders_ictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) > 0)[0]
    @property
    def neg_sig_responders_ictal(self):
        assert 'post' in self.expobj_exptype, f'incorrect call for {self.expobj_exptype} exp.'
        return np.where(np.nanmean(self.diff_responses_ictal[self.sig_units_ictal, :], axis=1) > 0)[0]

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
                                    colors=['green'], avg_with_std=True,
                                    suptitle=f"{self.expobj_id} - sig. responders",
                                    ylim=[-0.3, 0.5], fig=fig, ax=axs[0],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant positive responders.")
        if len(neg_avg_traces[0]) > 0:
            plot_periphotostim_avg2(dataset=neg_avg_traces, fps=expobj.fps,
                                    legend_labels=[f"neg. cells: {len(neg_avg_traces[0])}"],
                                    colors=['black'], avg_with_std=True,
                                    title=f"{self.expobj_id} - -ve sig. responders",
                                    ylim=[-0.3, 0.5], fig=fig, ax=axs[1],
                                    pre_stim_sec=PhotostimAnalysisSlmTargets._pre_stim_sec,
                                    show=False, fontsize='small', figsize=(4, 4),
                                    xlabel='Time (secs)', ylabel='Avg. Stim. Response (dF/stdF)')
        else:
            print(f"**** {expobj.t_series_name} has no statistically significant negative responders.")

        fig.show()


    # 2.2) PLOT -- BAR PLOT OF AVG MAGNITUDE OF RESPONSE

    @staticmethod
    def plot__avg_magnitude_response():
        """plot bar plot of avg magnitude of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0)
        def return__avg_magntiude_pos_response(**kwargs):
            """return avg magnitude of positive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_possig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        def return__avg_magntiude_pos_response_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_interictal)

        # @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        # def return__avg_magntiude_pos_response(**kwargs):
        #     """return avg magnitude of positive responders of stim trials - ictal"""
        #     expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        #     return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_ictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0)
        def return__avg_magntiude_neg_response(**kwargs):
            """return avg magnitude of negitive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.pre4ap_negsig_responders_responses)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        def return__avg_magntiude_neg_response_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_negsig_responders_responses_interictal)

        pos_baseline = return__avg_magntiude_pos_response()
        pos_interictal = return__avg_magntiude_pos_response_interictal()
        neg_baseline = return__avg_magntiude_neg_response()
        neg_interictal = return__avg_magntiude_neg_response_interictal()

        pplot.plot_bar_with_points(data=[pos_baseline, pos_interictal], paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['black', 'green'], y_label='Avg. magnitude of response', title='Positive responders')

        pplot.plot_bar_with_points(data=[neg_baseline, neg_interictal], paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['black', 'green'], y_label='Avg. magnitude of response', title='Negative responders')


    # 2.3) PLOT -- BAR PLOT OF AVG TOTAL NUMBER OF POS. AND NEG RESPONSIVE CELLS
    @staticmethod
    def plot__avg_num_responders():
        """plot bar plot of avg number of statistically significant responders across baseline and interictal, split up by positive and negative responders"""
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0)
        def return__num_pos_responders(**kwargs):
            """return avg magnitude of positive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.pre4ap_num_pos

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        def return__num_pos_responders_interictal(**kwargs):
            """return avg magnitude of positive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.post4ap_num_pos_interictal

        # @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        # def return__avg_magntiude_pos_response(**kwargs):
        #     """return avg magnitude of positive responders of stim trials - ictal"""
        #     expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        #     return np.mean(expobj.PhotostimResponsesNonTargets.post4ap_possig_responders_responses_ictal)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=0)
        def return__num_neg_responders(**kwargs):
            """return avg magnitude of negitive responders of stim trials -pre4ap """
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.pre4ap_num_neg

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=0)
        def return__num_neg_responders_interictal(**kwargs):
            """return avg magnitude of negitive responders of stim trials - interictal"""
            expobj: Union[alloptical, Post4ap] = kwargs['expobj']
            return expobj.PhotostimResponsesNonTargets.post4ap_num_neg_interictal

        pos_baseline = return__num_pos_responders()
        pos_interictal = return__num_pos_responders_interictal()
        neg_baseline = return__num_neg_responders()
        neg_interictal = return__num_neg_responders_interictal()

        pplot.plot_bar_with_points(data=[pos_baseline, pos_interictal], paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['black', 'green'], y_label='Number of responders', title='Positive responders')

        pplot.plot_bar_with_points(data=[neg_baseline, neg_interictal], paired=True, points = True, x_tick_labels=['baseline', 'interictal'],
                                   colors=['black', 'green'], y_label='Number of responders', title='Negative responders')



class PhotostimResponsesNonTargetsResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesNonTargets.pkl'

    def __init__(self):
        super().__init__()

        self.possig_responders_avgresponse = {'baseline': None, 'interictal': None, 'ictal': None}
        self.negsig_responders_avgresponse = {'baseline': None, 'interictal': None, 'ictal': None}

REMAKE = False
if not os.path.exists(PhotostimResponsesNonTargetsResults.SAVE_PATH) or REMAKE:
    results = PhotostimResponsesNonTargetsResults()
    results.save_results()

if __name__ == '__main__':

    # %%
    main = PhotostimResponsesQuantificationNonTargets
    results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

    # %%
    # main.run__initPhotostimResponsesQuantificationNonTargets()
    main.run_allopticalNontargets()  # <-- using as pipeline right now to test multiple functions for one exp
    # main.run__create_anndata()
    # main.collect__sig_responders_responses()

    main.plot__avg_magnitude_response()
    main.plot__avg_num_responders()


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
