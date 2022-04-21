from typing import Union, List, Dict

import seaborn as sns
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
import funcsforprajay.funcs as pj

from _utils_._anndata import AnnotatedData2
from _utils_.io import import_expobj

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# COLLECTING PHOTOSTIM TIMED DATA, PROCESSING AND ANALYSIS FOR SLM TARGETS TRACES

class PhotostimAnalysisSlmTargets(Quantification):
    """general photostim timed processing, analyses for SLM targets"""

    save_path = SAVE_LOC + 'PhotostimAnalysisSlmTargets.pkl'
    valid_targets_trace_types = ['trace_dFF', 'raw']
    # _pre_stim_sec = 1
    # _post_stim_sec = 3
    # pre_stim_response_window_msec = 500 # msec
    # post_stim_response_window_msec = 500  # msec

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimAnalysisSlmTargets MODULE to expobj: {expobj.t_series_name}')
        self.create_anndata(expobj=expobj)
        # self._fps = expobj.fps

    # @property
    # def pre_stim_fr(self):
    #     return int(self._pre_stim_sec * self._fps)  # length of pre stim trace collected (in frames)
    #
    # @property
    # def post_stim_fr(self):
    #     return int(self._post_stim_sec * self._fps)  # length of post stim trace collected (in frames)
    #
    # @property
    # def pre_stim_response_frames_window(self):
    #     return int(self._fps * self.pre_stim_response_window_msec / 1000)  # length of the pre stim response test window (in frames)
    #
    # @property
    # def post_stim_response_frames_window(self):
    #     return int(self._fps * self.post_stim_response_window_msec / 1000)  # length of the post stim response test window (in frames)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
    def run__init(**kwargs):
        expobj = kwargs['expobj']
        expobj.PhotostimAnalysisSlmTargets = PhotostimAnalysisSlmTargets(expobj=expobj)
        expobj.save()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
    def run__add_fps(**kwargs):
        expobj = kwargs['expobj']
        expobj.PhotostimAnalysisSlmTargets._fps = expobj.fps
        expobj.save()


    def __repr__(self):
        return f"PhotostimAnalysisSlmTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    def create_anndata(self, expobj):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial.

        """

        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from SLM targets information
        obs_meta = pd.DataFrame(
            columns=['SLM group #', 'SLM target coord'], index=range(expobj.n_targets_total))
        for target_idx, coord in enumerate(expobj.target_coords_all):
            for groupnum, coords in enumerate(expobj.target_coords):
                if coord in coords:
                    obs_meta.loc[target_idx, 'SLM group #'] = groupnum
                    obs_meta.loc[target_idx, 'SLM target coord'] = coord
                    break

        # build numpy array for multidimensional obs metadata
        obs_m = {'SLM targets areas': []}
        for target, areas in enumerate(expobj.target_areas):
            obs_m['SLM targets areas'].append(np.asarray(areas))
        obs_m['SLM targets areas'] = np.asarray(obs_m['SLM targets areas'])

        # SETUP THE VARIABLES ANNOTATIONS TO USE IN anndata
        # build dataframe for var annot's - based on stim_start_frames
        var_meta = pd.DataFrame(index=['im_time_secs', 'stim_start_frame', 'wvfront in sz', 'seizure location'],
                                columns=range(len(expobj.stim_start_frames)))
        for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
            if 'pre' in expobj.exptype:
                var_meta.loc['wvfront in sz', fr_idx] = None
                var_meta.loc['seizure location', fr_idx] = None
            elif 'post' in expobj.exptype:
                if stim_frame in expobj.stimsWithSzWavefront:
                    var_meta.loc['wvfront in sz', fr_idx] = True
                    var_meta.loc['seizure location', fr_idx] = (
                        expobj.stimsSzLocations.coord1[stim_frame], expobj.stimsSzLocations.coord2[stim_frame])
                else:
                    var_meta.loc['wvfront in sz', fr_idx] = False
                    var_meta.loc['seizure location', fr_idx] = None
            var_meta.loc['stim_start_frame', fr_idx] = stim_frame
            var_meta.loc['im_time_secs', fr_idx] = stim_frame / expobj.fps

        # SET PRIMARY DATA
        assert hasattr(expobj, 'PhotostimResponsesSLMTargets'), 'no photostim responses found to use to create anndata base.'
        print(f"\t\----- CREATING annotated data object using AnnData:")
        # create anndata object
        photostim_responses_adata = AnnotatedData2(X=expobj.PhotostimResponsesSLMTargets.adata.X,
                                                   obs=obs_meta, var=var_meta.T, obsm=obs_m,
                                                   data_label=expobj.PhotostimResponsesSLMTargets.adata.data_label)

        print(f"Created: {photostim_responses_adata}")
        self.adata = photostim_responses_adata


    # 0) COLLECT ALL PHOTOSTIM TIMED TRACE SNIPPETS --> found under PhotostimResponsesQuantificationSLMtargets class

    # 1) plot peri-photostim avg traces for all trials analyzed to make sure they look alright -- plot as little postage stamps
    @staticmethod
    @Utils.run_for_loop_across_exps(run_trials=[])#[AllOpticalExpsToAnalyze.pre_4ap_trials[0][0], AllOpticalExpsToAnalyze.post_4ap_trials[0][0]])
    def plot_postage_stamps_photostim_traces(to_plot='dFF',**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']

        if to_plot == 'delta dF':
            responses = expobj.responses_SLMtargets_tracedFF
            hits = expobj.hits_SLMtargets_tracedFF
            trace_snippets = expobj.SLMTargets_tracedFF_stims_dff  # TODO confirm that the pre-stim period has mean of 0 for all these traces!
            stimsuccessrates = expobj.StimSuccessRate_SLMtargets_tracedFF
            y_label = 'delta dF'
        elif to_plot == 'dFF':
            responses = expobj.responses_SLMtargets
            hits = expobj.hits_SLMtargets
            trace_snippets = expobj.SLMTargets_stims_dff
            stimsuccessrates = expobj.StimSuccessRate_SLMtargets
            y_label = '% dFF'
        else:
            raise ValueError('must provide to_plot as either `dFF` or `delta dF`')

        responses_magnitudes_successes = {}
        response_traces_successes = {}
        responses_magnitudes_failures = {}
        response_traces_failures = {}

        nrows = expobj.n_targets_total // 4
        if expobj.n_targets_total % 4 > 0:
            nrows += 1
        ncols = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3),
                                constrained_layout=True)
        counter = 0
        axs[0, 0].set_xlabel('Time (secs)')
        axs[0, 0].set_ylabel(y_label)

        for cell in range(trace_snippets.shape[0]):
            a = counter // 4
            b = counter % 4
            alpha = 1 * (stimsuccessrates[cell] / 100)
            print(f'plotting target #{counter}, success rate: {stimsuccessrates[cell]}\n')
            if cell not in responses_magnitudes_successes.keys():
                responses_magnitudes_successes[cell] = []
                response_traces_successes[cell] = np.zeros((trace_snippets.shape[-1]))
                responses_magnitudes_failures[cell] = []
                response_traces_failures[cell] = np.zeros((trace_snippets.shape[-1]))

            success_stims = np.where(hits.loc[cell] == 1)
            fail_stims = np.where(hits.loc[cell] == 0)

            x_range = np.linspace(0, len(trace_snippets[cell][0]) / expobj.fps, len(trace_snippets[cell][0]))

            # success_stims = np.where(expobj.responses_SLMtargets_dfprestimf.loc[cell] >= 0.1 * 100)
            # fail_stims = np.where(expobj.responses_SLMtargets_dfprestimf.loc[cell] < 0.1 * 100)
            for i in success_stims[0]:
                trace = trace_snippets[cell][i]
                axs[a, b].plot(x_range, trace, color='skyblue', zorder=2, alpha=0.05)

            for i in fail_stims[0]:
                trace = trace_snippets[cell][i]
                axs[a, b].plot(x_range, trace, color='gray', zorder=3, alpha=0.05)

            if len(success_stims[0]) > 5:
                success_avg = np.nanmean(trace_snippets[cell][success_stims], axis=0)
                axs[a, b].plot(x_range, success_avg, color='navy', linewidth=2, zorder=4, alpha=1)
            if len(fail_stims[0]) > 5:
                failures_avg = np.nanmean(trace_snippets[cell][fail_stims], axis=0)
                axs[a, b].plot(x_range, failures_avg, color='black', linewidth=2, zorder=4, alpha=1)
            axs[a, b].set_ylim([-0.2 * 100, 1.0 * 100])
            axs[a, b].text(0.98, 0.97, f">10% dFF: {stimsuccessrates[cell]:.0f}%",
                           verticalalignment='top', horizontalalignment='right',
                           transform=axs[a, b].transAxes, fontweight='bold',
                           color='black')
            axs[a, b].margins(0)
            axs[a, b].axvspan(expobj.pre_stim / expobj.fps, (expobj.pre_stim + expobj.stim_duration_frames) / expobj.fps, color='mistyrose',
                              zorder=0)

            counter += 1
        fig.suptitle(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - {len(trace_snippets)} targets",
                     y = 0.995)
        # fig.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s_individual targets dFF.png' % (date, j[:-6], date, j))
        fig.tight_layout(pad=1.8)
        fig.show()

    # 1.1) plot peri-photostim avg traces across all targets for all experiment trials
    def plot_peristim_avg_photostims(self):

        for trial in pj.flattenOnce(AllOpticalExpsToAnalyze.pre_4ap_trials):
            if 'RL109' in trial:
                from _utils_.io import import_expobj
                expobj: alloptical = import_expobj(exp_prep=trial)
                trace_snippets_avg = np.mean(expobj.SLMTargets_stims_dff, axis=1)
                print(trace_snippets_avg.shape[1])
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[4, 8])
                # aoplot.plot_periphotostim_avg2(dataset=trace_snippets_avg, fps=expobj.fps, pre_stim_sec=expobj.PhotostimAnalysisSlmTargets._pre_stim_sec, title=f'{expobj.t_series_name}')
                from _utils_.alloptical_plotting import plot_periphotostim_avg
                plot_periphotostim_avg(arr=trace_snippets_avg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                              title=f'{expobj.t_series_name} - pre4ap', expobj=expobj,
                                              x_label='Time (secs)', y_label='dFF response', fig=fig, ax=axs[0],
                                              show=False)

                post4ap_exp = AllOpticalExpsToAnalyze.find_matched_trial(pre4ap_trial_name=expobj.t_series_name)
                print(post4ap_exp)
                expobj: Post4ap = import_expobj(exp_prep=post4ap_exp)
                trace_snippets_avg = np.mean(expobj.SLMTargets_stims_dff, axis=1)
                print(trace_snippets_avg.shape[1], '\n')
                plot_periphotostim_avg(arr=trace_snippets_avg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                              title=f'{expobj.t_series_name} - post4ap', expobj=expobj,
                                              x_label='Time (secs)', y_label='dFF response', fig=fig, ax=axs[1])
                fig.show()

    # 1.2) plot photostim avg of all targets from each experiment
    @staticmethod
    def plot__avg_photostim_dff_allexps():

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[4, 8])
        axs[0].set_ylim([-15, 50])
        axs[1].set_ylim([-15, 50])
        axs[0].axvspan(0, 0.250, alpha=1, color='tomato', zorder=5)
        axs[1].axvspan(0, 0.250, alpha=1, color='tomato', zorder=5)

        for ax in axs:
            ax.set_xlabel('Time (secs)')
        for ax in axs:
            ax.set_ylabel('dFF')
        axs[0].set_title('baseline')
        axs[1].set_title('interictal')


        for exp in AllOpticalExpsToAnalyze.exp_ids:
            for trial in pj.flattenOnce(AllOpticalExpsToAnalyze.pre_4ap_trials):
                if exp in trial:
                    # pre4ap trial
                    expobj: alloptical = import_expobj(exp_prep=trial)
                    trace_snippets_avg = np.mean(expobj.SLMTargets_tracedFF_stims_dffAvg, axis=0)

                    pre_stim_slice = np.s_[0: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr]
                    stim_dur_slice = np.s_[expobj.PhotostimAnalysisSlmTargets.pre_stim_fr: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + int(0.250 * expobj.fps)]
                    post_stim_slice = np.s_[expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + expobj.stim_duration_frames: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + expobj.stim_duration_frames + expobj.PhotostimAnalysisSlmTargets.post_stim_fr]

                    dff_trace = np.concatenate((trace_snippets_avg[pre_stim_slice], np.array([1000] * int(0.250 * expobj.fps)), trace_snippets_avg[post_stim_slice]))

                    pre_stim_x = np.linspace(-expobj.PhotostimAnalysisSlmTargets._pre_stim_sec, 0, int(expobj.PhotostimAnalysisSlmTargets._pre_stim_sec * expobj.fps))  # x scale, but in time domain (transformed from frames based on the provided fps)
                    stim_dur_x = np.linspace(0, 0.250, int(0.250 * expobj.fps))
                    post_stim_x = np.linspace(0.250, 0.250 + expobj.PhotostimAnalysisSlmTargets._post_stim_sec, int(expobj.PhotostimAnalysisSlmTargets._post_stim_sec * expobj.fps))  # x scale, but in time domain (transformed from frames based on the provided fps)

                    x_scale = np.concatenate((pre_stim_x, stim_dur_x, post_stim_x))
                    assert len(x_scale) == len(dff_trace), 'x axis scale is too short or too long.'
                    axs[0].plot(x_scale, dff_trace, color='black')


                    # post4ap trial
                    post4ap_exp = AllOpticalExpsToAnalyze.find_matched_trial(pre4ap_trial_name=expobj.t_series_name)
                    print(post4ap_exp)
                    expobj: Post4ap = import_expobj(exp_prep=post4ap_exp)

                    trace_snippets_avg = np.mean(expobj.SLMTargets_tracedFF_stims_dffAvg_outsz, axis=0)

                    pre_stim_slice = np.s_[0: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr]
                    stim_dur_slice = np.s_[expobj.PhotostimAnalysisSlmTargets.pre_stim_fr: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + int(0.250 * expobj.fps)]
                    post_stim_slice = np.s_[expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + expobj.stim_duration_frames: expobj.PhotostimAnalysisSlmTargets.pre_stim_fr + expobj.stim_duration_frames + expobj.PhotostimAnalysisSlmTargets.post_stim_fr]

                    dff_trace = np.concatenate((trace_snippets_avg[pre_stim_slice], np.array([1000] * int(0.250 * expobj.fps)), trace_snippets_avg[post_stim_slice]))

                    pre_stim_x = np.linspace(-expobj.PhotostimAnalysisSlmTargets._pre_stim_sec, 0, int(expobj.PhotostimAnalysisSlmTargets._pre_stim_sec * expobj.fps))  # x scale, but in time domain (transformed from frames based on the provided fps)
                    stim_dur_x = np.linspace(0, 0.250, int(0.250 * expobj.fps))
                    post_stim_x = np.linspace(0.250, 0.250 + expobj.PhotostimAnalysisSlmTargets._post_stim_sec, int(expobj.PhotostimAnalysisSlmTargets._post_stim_sec * expobj.fps))  # x scale, but in time domain (transformed from frames based on the provided fps)

                    x_scale = np.concatenate((pre_stim_x, stim_dur_x, post_stim_x))
                    assert len(x_scale) == len(dff_trace), 'x axis scale is too short or too long.'
                    axs[1].plot(x_scale, dff_trace, color='green')




        fig.show()


    # 2) calculating mean variability across targets:
    def calculate_variability(self, stims):
        """calculate standard deviation (ddof = 1) variability of photostim repsonses across trials for each inidividual target."""

        assert hasattr(self, 'adata'), 'cannot find .adata'
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        print('calculating variability of photostim responses.')

        std_vars = []
        for target in self.adata.obs.index:
            target = int(target)
            std_var = np.std(self.adata.X[target, stims], ddof=1)
            std_vars.append(std_var)
        return std_vars

    @staticmethod
    def plot__variability():
    # 1) plotting mean photostim response magnitude across experiments and experimental groups
        """create plot of mean photostim responses magnitudes for all three exp groups"""

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                        allow_rerun=1)
        def pre4apexps_collect_photostim_stdvars(**kwargs):
            expobj: alloptical = kwargs['expobj']
            if 'pre' in expobj.exptype:
                # all stims
                photostim_stdvars = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims='all')
                return np.mean(photostim_stdvars)

        photostim_stdvars_baseline = pre4apexps_collect_photostim_stdvars()

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False,
                                        allow_rerun=1)
        def post4apexps_collect_photostim_stdvars(**kwargs):
            expobj: Post4ap = kwargs['expobj']
            if 'post' in expobj.exptype:
                # interictal stims
                mean_photostim_stdvars_interictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_outsz)

                # ictal stims
                mean_photostim_stdvars_ictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_insz)

                return np.mean(mean_photostim_stdvars_interictal), np.mean(mean_photostim_stdvars_ictal)

        func_collector = post4apexps_collect_photostim_stdvars()

        if len(func_collector) > 0:
            photostim_stdvars_interictal, photostim_stdvars_ictal = np.asarray(func_collector)[:,
                                                                                  0], np.asarray(
                func_collector)[:, 1]

            pplot.plot_bar_with_points(
                data=[photostim_stdvars_baseline, photostim_stdvars_interictal,
                      photostim_stdvars_ictal],
                x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False, colors=['blue', 'green', 'purple'],
                expand_size_x=0.4, title='Average stnd dev.', y_label='standard dev. (% dFF)')

            return photostim_stdvars_baseline, photostim_stdvars_interictal, photostim_stdvars_ictal

    @staticmethod
    def plot__schematic_variability_measurement():
        """plot for schematic of calculating variability"""
        amplitudes = np.random.rand(50)

        fig, ax = plt.subplots(figsize=[2, 2])
        total_x = []
        total_y = []
        for i in amplitudes:
            x = []
            y = []

            flat_x = np.linspace(-5, -1.5, 10)
            flat_y = [(np.random.rand(1)[0]) / 2 for x in flat_x]
            x.extend(flat_x)
            y.extend(flat_y)

            # exp_rise_x = np.linspace(-2.5, -1.5 + i/5, 10)
            # exp_rise_y = np.exp(exp_rise_x + 3)
            # exp_rise_y = [(y + np.random.rand(1)[0]/2) for y in exp_rise_y]
            # x.extend(exp_rise_x)
            # y.extend(exp_rise_y)

            exp_decay_x = np.linspace(-1.5 + i / 5, 7 + i / 5, 10)
            exp_decay_y = np.exp(-exp_decay_x / 1.3 - (i * 2))
            exp_decay_y = [(y + np.random.rand(1)[0] / 2) for y in exp_decay_y]
            x.extend(exp_decay_x)
            y.extend(exp_decay_y)

            # end_x = np.linspace(3+i/5, 7, 30)
            # end_y = [(np.random.rand(1)[0])/2 for x in end_x]
            # x.extend(end_x)
            # y.extend(end_y)

            ax.plot(x, y, lw=2, color='forestgreen', alpha=0.25)
            total_x.append(x)
            total_y.append(y)

        mean = np.mean(total_y, axis=0)
        std = np.std(total_y, axis=0, ddof=1)

        ax.plot(np.mean(total_x, axis=0), np.mean(total_y, axis=0), lw=2, color='black')
        # ax.fill_between(np.mean(total_x, axis=0), mean-std, mean+std, color='gray')

        fig.show()

    # 2.2) plot mean magnitude of photostim responses vs. variability of photostim responses
    @staticmethod
    def plot__mean_response_vs_variability():
        """create plot of mean photostim responses vs. variability for all three exp groups"""

        fig, ax = plt.subplots(ncols=3)


        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                        allow_rerun=1)
        def pre4apexps_collect_photostim_stdvars(**kwargs):
            expobj: alloptical = kwargs['expobj']
            if 'pre' in expobj.exptype:
                # all stims
                photostim_stdvars = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims='all')
                return photostim_stdvars


        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                        allow_rerun=1)
        def pre4apexps_collect_photostim_responses(**kwargs):
            expobj: alloptical = kwargs['expobj']
            if 'pre' in expobj.exptype:
                # all stims
                mean_photostim_responses = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                    stims='all')
                return mean_photostim_responses

        mean_photostim_responses_baseline = pre4apexps_collect_photostim_responses()
        photostim_stdvars_baseline = pre4apexps_collect_photostim_stdvars()

        pplot.make_general_scatter([pj.flattenOnce(mean_photostim_responses_baseline)],
                                   [pj.flattenOnce(photostim_stdvars_baseline)], x_labels=['Avg. Photostim Response (% dFF)'],
                                   y_labels = ['Avg. standard dev. (% dFF)'], ax_titles=['Targets: avg. response vs. variability - baseline'],
                                   facecolors=['none'], edgecolors=['royalblue'], lw=1.3, figsize=(4,4))


        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False,
                                        allow_rerun=1)
        def post4apexps_collect_photostim_stdvars(**kwargs):
            expobj: Post4ap = kwargs['expobj']
            if 'post' in expobj.exptype:
                # interictal stims
                mean_photostim_stdvars_interictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_outsz)

                # ictal stims
                mean_photostim_stdvars_ictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_insz)

                return mean_photostim_stdvars_interictal, mean_photostim_stdvars_ictal

        func_collector_std = post4apexps_collect_photostim_stdvars()

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False,
                                        allow_rerun=1)
        def post4apexps_collect_photostim_responses(**kwargs):
            expobj: Post4ap = kwargs['expobj']
            if 'post' in expobj.exptype:
                # interictal stims
                mean_photostim_responses_interictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                    stims=expobj.stim_idx_outsz)

                # ictal stims
                mean_photostim_responses_ictal = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                    stims=expobj.stim_idx_insz)

                return mean_photostim_responses_interictal, mean_photostim_responses_ictal

        func_collector_responses = post4apexps_collect_photostim_responses()

        mean_photostim_responses_interictal, mean_photostim_responses_ictal = np.asarray(func_collector_responses)[:,
                                                                              0], np.asarray(func_collector_responses)[:, 1]

        photostim_stdvars_interictal, photostim_stdvars_ictal = np.asarray(func_collector_std)[:,
                                                                              0], np.asarray(func_collector_std)[:, 1]

        pplot.make_general_scatter([pj.flattenOnce(mean_photostim_responses_interictal)],
                                   [pj.flattenOnce(photostim_stdvars_interictal)], x_labels=['Avg. Photostim Response (% dFF)'],
                                   y_labels = ['Avg. standard dev. (%dFF)'], ax_titles=['Targets: avg. response vs. variability - interictal'],
                                   facecolors=['none'], edgecolors=['seagreen'], lw=1.3, figsize=(4,4))


        pplot.make_general_scatter([pj.flattenOnce(mean_photostim_responses_ictal)],
                                   [pj.flattenOnce(photostim_stdvars_ictal)], x_labels=['Avg. Photostim Response (% dFF)'],
                                   y_labels = ['Avg. standard dev. (% dFF)'], ax_titles=['Targets: avg. response vs. variability - ictal'],
                                   facecolors=['none'], edgecolors=['mediumorchid'], lw=1.3, figsize=(4,4))





if __name__ == '__main__':
    main = PhotostimAnalysisSlmTargets
    main.run__init()
    main.run__add_fps()

    # RUNNING PLOTS:
    # main.plot_postage_stamps_photostim_traces()
    # main.plot__variability()
    # main.plot__mean_response_vs_variability()
    main.plot__avg_photostim_dff_allexps()



# expobj = import_expobj(prep='RL108', trial='t-009')



# %%


