from typing import Union, List, Dict

import seaborn as sns
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import variation, mannwhitneyu, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import _alloptical_utils as Utils
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesQuantificationSLMtargets, \
    PhotostimResponsesSLMtargetsResults
from _analysis_._utils import Quantification, Results
from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
import funcsforprajay.funcs as pj

from _utils_._anndata import AnnotatedData2
from _utils_.alloptical_plotting import plot_settings
from _utils_.io import import_expobj

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-traces/"

results = PhotostimResponsesSLMtargetsResults.load()

# COLLECTING PHOTOSTIM TIMED DATA, PROCESSING AND ANALYSIS FOR SLM TARGETS TRACES

#!/usr/bin/env python3
def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

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
        expobj: alloptical = kwargs['expobj']
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

    @staticmethod
    def collect_all_targets_all_exps(run_pre4ap_trials=0, run_post4ap_trials=True):
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=run_pre4ap_trials, allow_rerun=1)
        def baseline_targets_responses(**kwargs):
            expobj = kwargs['expobj']
            if np.nanmean(expobj.PhotostimAnalysisSlmTargets.adata.X) > 10:
            # print(np.nanmean(expobj.PhotostimAnalysisSlmTargets.adata.X))
                return expobj.PhotostimAnalysisSlmTargets.adata.X, expobj.t_series_name
        if run_pre4ap_trials:
            func_collector = baseline_targets_responses()
            responses, expids = np.asarray(func_collector)[:, 0].tolist(), np.asarray(func_collector)[:, 1]

            num_targets = 0
            num_stims = 100
            for response in responses:
                num_targets += response.shape[0]
                if response.shape[1] > num_stims:
                    num_stims = response.shape[1]

            all_responses = np.empty(shape=(num_targets, num_stims))
            all_responses[:] = np.NaN
            all_exps = []
            num_targets = 0
            for response, expid in zip(responses, expids):
                targets = response.shape[0]
                stims = response.shape[1]
                all_responses[num_targets: num_targets + targets, :stims] = response
                all_exps.extend([expid]*targets)
                num_targets += targets

            baseline_adata = AnnotatedData2(all_responses, obs ={'exp': all_exps}, var={'stim_idx': range(all_responses.shape[1])})
            results.baseline_adata = baseline_adata
            results.save_results()


        @Utils.run_for_loop_across_exps(run_post4ap_trials=run_post4ap_trials, allow_rerun=1)
        def interictal_targets_responses(**kwargs):
            expobj: Post4ap = kwargs['expobj']
            stims = expobj.stim_idx_outsz
            pre4ap = AllOpticalExpsToAnalyze.find_matched_trial(post4ap_trial_name=expobj.t_series_name)
            pre4ap_expobj: alloptical = import_expobj(exp_prep=pre4ap)

            if pre4ap_expobj.t_series_name in tuple(results.baseline_adata.obs['exp']):
                # print(np.mean(expobj.PhotostimResponsesSLMTargets.adata.layers['dFF (zscored)'][:, stims]))
                return expobj.PhotostimResponsesSLMTargets.adata.layers['dFF (zscored)'][:, stims], expobj.t_series_name

        if run_post4ap_trials:
            func_collector = interictal_targets_responses()
            responses, expids = np.asarray(func_collector)[:, 0].tolist(), np.asarray(func_collector)[:, 1]

            num_targets = 0
            num_stims = 100
            for response in responses:
                num_targets += response.shape[0]
                if response.shape[1] > num_stims:
                    num_stims = response.shape[1]

            all_responses = np.empty(shape=(num_targets, num_stims))
            all_responses[:] = np.NaN
            all_exps = []
            num_targets = 0
            for response, expid in zip(responses, expids):
                targets = response.shape[0]
                stims = response.shape[1]
                all_responses[num_targets: num_targets + targets, :stims] = response
                all_exps.extend([expid]*targets)
                num_targets += targets

            interictal_adata = AnnotatedData2(all_responses, obs ={'exp': all_exps}, var={'stim_idx': range(all_responses.shape[1])}, data_label='z scored (to baseline)')
            results.interictal_adata = interictal_adata
            results.save_results()



    # 0) COLLECT ALL PHOTOSTIM TIMED TRACE SNIPPETS --> found under PhotostimResponsesQuantificationSLMtargets class

    # 1) plot peri-photostim avg traces for all targets from all exp analyzed to make sure they look alright -- plot as little postage stamps
    @staticmethod
    @Utils.run_for_loop_across_exps(run_trials=[AllOpticalExpsToAnalyze.pre_4ap_trials[0][0], AllOpticalExpsToAnalyze.post_4ap_trials[0][0]], allow_rerun=1)
    def plot_postage_stamps_photostim_traces(to_plot='delta dF',**kwargs):
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


    @staticmethod
    @Utils.run_for_loop_across_exps(run_trials=[
        AllOpticalExpsToAnalyze.pre_4ap_trials[0][0],
                                                AllOpticalExpsToAnalyze.post_4ap_trials[0][0]], allow_rerun=1)
    def plot_variability_photostim_traces_by_targets(to_plot='dFF',**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']

        if to_plot == 'delta dF':
            responses = expobj.responses_SLMtargets_tracedFF
            hits = expobj.hits_SLMtargets_tracedFF
            trace_snippets = expobj.SLMTargets_tracedFF_stims_dff  # TODO confirm that the pre-stim period has mean of 0 for all these traces!
            stimsuccessrates = expobj.StimSuccessRate_SLMtargets_tracedFF
            y_label = 'delta dF'
        elif to_plot == 'dFF':
            responses = expobj.PhotostimAnalysisSlmTargets.adata.X
            hits = expobj.hits_SLMtargets
            trace_snippets = expobj.SLMTargets_stims_dff
            stimsuccessrates = expobj.StimSuccessRate_SLMtargets
            y_label = '% dFF'
        else:
            raise ValueError('must provide to_plot as either `dFF` or `delta dF`')

        # choose 10 random cells from each dataset
        choice = np.random.choice(a=np.arange(0, trace_snippets.shape[0]), size=10, replace=False)

        total_plot = len(choice)
        # total_plot = expobj.n_targets_total


        # nrows = expobj.n_targets_total // 4
        ncols = 10
        nrows = total_plot // ncols
        if total_plot % ncols > 0 or total_plot % ncols == 0:
            nrows += 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.2, nrows*1.75),
                                constrained_layout=True, dpi=600)
        counter = 0
        axs[0, 0].set_xlabel('Time (secs)')
        axs[0, 0].set_ylabel(y_label)

        import matplotlib as mpl
        cmap = plt.cm.get_cmap('viridis')
        # for cell in range(trace_snippets.shape[0]):  # for plotting all cells
        for cell in choice:  # for plotting only randomly chosen cells
            a = counter // ncols
            b = counter % ncols
            alpha = 1 * (stimsuccessrates[cell] / 100)
            responses_ = responses[cell]

            cv = np.std(responses_, ddof=1)/np.mean(responses_)

            print(f'plotting target #{counter}, CV: {cv:.3f}\n')

            x_range = np.linspace(0, len(trace_snippets[cell][0]) / expobj.fps, len(trace_snippets[cell][0]))


            # ax.scatter(np.random.choice(x_range[20:-20], size=len(responses_)), responses_, s=20, zorder=1, alpha=0.3)
            x = simple_beeswarm(responses_)
            # transform x
            x_new = [(i + 1) / 2 * (x_range[-1]) for i in x]
            # axs[a,b].scatter(x_new, responses_, s=10, alpha=0.4, color='yellowgreen', zorder=1)
            sc = axs[a,b].scatter(x_new, responses_, s=10, alpha=0.4, c=[cv]*len(responses_), zorder=1, cmap='viridis',
                                  vmin=0, vmax=3)
            axs[a,b].set_xlim([-1.5, x_range[-1] + 1.5])
            cbar = plt.colorbar(sc)
            cbar.remove()

            ax = axs[a, b].twiny()
            avg = np.nanmean(trace_snippets[cell], axis=0)
            ax.plot(x_range[:-4], pj.smoothen_signal(avg, 5), color='black', linewidth=2, zorder = 5)



            ax.set_ylim([-50,200])
            # axs[a, b].set_ylim([-0.2 * 100, 2.0 * 100])
            # axs[a, b].text(0.98, 0.97, f"{cv:.1f}",
            #                verticalalignment='top', horizontalalignment='right',
            #                transform=axs[a, b].transAxes, fontweight='bold', fontsize=8,
            #                color='black')
            axs[a, b].margins(0)
            axs[a, b].spines['top'].set_visible(False)
            axs[a, b].spines['right'].set_visible(False)
            axs[a, b].spines['bottom'].set_visible(False)
            axs[a, b].spines['left'].set_visible(False)
            axs[a, b].set_xticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # axs[a, b].axvspan(expobj.pre_stim / expobj.fps, (expobj.pre_stim + expobj.stim_duration_frames) / expobj.fps, color='mistyrose',
            #                   zorder=0)
            counter += 1
        fig.suptitle(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - {len(trace_snippets)} targets",
                     y = 0.995)
        fig.tight_layout(pad=0.2)
        fig.show()
        save_path_full = f'{SAVE_FIG}/photostim-variation-individual-targets-dFF-{expobj.t_series_name}-{expobj.exptype}.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)

        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
        fig, ax = plt.subplots(figsize=(6, 1), dpi=600)
        ax.imshow([colors], extent=[0, 10, 0, 1])
        fig.tight_layout(pad=0.4)
        fig.show()
        save_path_full = f'{SAVE_FIG}/photostim-variation-individual-targets-dFF-colorbar.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)

    # 8) correlation matrix of photostimulation targets
    @staticmethod
    def correlation_matrix_all_targets():
        results = PhotostimResponsesSLMtargetsResults.load()

        # BASELINE
        responses = results.baseline_adata.X
        z_scored_responses = stats.zscore(responses, axis=1)  # z scoring of targets responses to their own distribution

        # correlation of targets
        fig, ax = plt.subplots(figsize=(3,3), dpi=400)
        targets_corr_mat = np.corrcoef(z_scored_responses)
        mesh = ax.imshow(targets_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)

        # add lines for exps:
        colors = pj.make_random_color_array(len(np.unique(results.baseline_adata.obs['exp'])))
        for i, exp in enumerate(np.unique(results.baseline_adata.obs['exp'])):
            sameexp = np.where(results.baseline_adata.obs['exp'] == exp)[0]
            ax.plot(sameexp, [-10] * len(sameexp), lw=2, color = colors[i], solid_capstyle = 'projecting')
            # plt.plot(sameexp, [-10] * len(sameexp), lw=2, color = colors[i], solid_capstyle = 'projecting')
        # plt.show()

        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.suptitle(f'baseline - correlation across - targets', wrap=True)
        fig.tight_layout(pad=0.5)
        fig.show()
        save_path_full = f'{SAVE_FIG}/correlation_matrix_all_targets_all_exps_baseline.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)



        # correlation of stims
        fig, ax = plt.subplots(figsize=(3,3), dpi=400)
        stim_corr_mat = np.corrcoef(z_scored_responses.T)
        mesh = ax.imshow(stim_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)
        fig.suptitle(f'baseline - correlation across - stims', wrap=True)
        fig.tight_layout(pad=0.5)
        fig.show()
        save_path_full = f'{SAVE_FIG}/correlation_matrix_all_stims_all_exps_baseline.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)


        # INTERICTAL
        z_scored_responses = results.interictal_adata.X
        z_scored_responses = pd.DataFrame(z_scored_responses).iloc[:, :43]
        # z_scored_responses = stats.zscore(responses, axis=1)  # z scoring of targets responses to their own distribution

        # correlation of targets
        fig, ax = plt.subplots(figsize=(3,3), dpi=400)
        targets_corr_mat = np.corrcoef(z_scored_responses)
        # targets_corr_mat = z_scored_responses.T.corr()
        mesh = ax.imshow(targets_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)

        # add lines for exps:
        for i, exp in enumerate(np.unique(results.baseline_adata.obs['exp'])):
            sameexp = np.where(results.baseline_adata.obs['exp'] == exp)[0]
            ax.plot(sameexp, [-10] * len(sameexp), lw=2, color = colors[i], solid_capstyle = 'projecting')
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fig.suptitle(f'interictal - correlation across - targets', wrap=True)
        fig.tight_layout(pad=0.5)
        fig.show()
        save_path_full = f'{SAVE_FIG}/correlation_matrix_all_targets_all_exps_interictal.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)

        # correlation of stims
        fig, ax = plt.subplots(figsize=(3,3), dpi=400)
        stim_corr_mat = np.corrcoef(z_scored_responses.T)
        mesh = ax.imshow(stim_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)
        fig.suptitle(f'interictal - correlation across - stims', wrap=True)
        fig.tight_layout(pad=0.5)
        fig.show()
        save_path_full = f'{SAVE_FIG}/correlation_matrix_all_stims_all_exps_interictal.png'
        os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
        fig.savefig(save_path_full)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_trials=[AllOpticalExpsToAnalyze.pre_4ap_trials[0][0], AllOpticalExpsToAnalyze.post_4ap_trials[0][0]], allow_rerun=1)
    def correlation_matrix_responses(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        z_scored_responses = stats.zscore(expobj.PhotostimAnalysisSlmTargets.adata.X, axis=1)

        # correlation of targets
        fig, ax = plt.subplots(figsize=(3,3), dpi=300)
        targets_corr_mat = np.corrcoef(z_scored_responses)
        mesh = ax.imshow(targets_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)
        fig.suptitle(f'{expobj.t_series_name} - targets')
        fig.tight_layout(pad=0.5)
        fig.show()

        # correlation of stims
        fig, ax = plt.subplots(figsize=(3,3), dpi=300)
        stim_corr_mat = np.corrcoef(z_scored_responses.T)
        mesh = ax.imshow(stim_corr_mat, cmap='bwr')
        mesh.set_clim(-1, 1)
        cbar = fig.colorbar(mesh, boundaries=np.linspace(-0.5, 0.5, 1000), ticks=[-0.5, 0, 0.5], fraction=0.045)
        fig.suptitle(f'{expobj.t_series_name} - stims')
        fig.tight_layout(pad=0.5)
        fig.show()

    # 8.1) PCA of stims
    @staticmethod
    @Utils.run_for_loop_across_exps(run_trials=[AllOpticalExpsToAnalyze.pre_4ap_trials[0][0], AllOpticalExpsToAnalyze.post_4ap_trials[0][0]], allow_rerun=1)
    def pca_responses(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        z_scored_responses = stats.zscore(expobj.PhotostimAnalysisSlmTargets.adata.X, axis=1)
        mean_responses = np.mean(expobj.PhotostimAnalysisSlmTargets.adata.X, axis=1)
        variation_responses = variation(expobj.PhotostimAnalysisSlmTargets.adata.X, axis=1)

        n_comp = 10
        pca = PCA(n_components=n_comp)
        pca_responses = pca.fit_transform(z_scored_responses)

        # clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(z_scored_responses)
        kmeans.labels_

        # plot PCA transformed responses
        fig, ax = plt.subplots(figsize=(5, 4))
        # ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, color='green', edgecolor='yellowgreen', linewidth=3)
        # sc = ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, c=kmeans.labels_, cmap=plt.cm.get_cmap('viridis', 2))
        # fig.colorbar(sc, ticks=range(1), label ='kmeans clusters')
        sc = ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, c=variation_responses, cmap=plt.cm.get_cmap('viridis', 10), vmin=0, vmax=3)
        fig.colorbar(sc, ticks=range(10), label ='variation responses')
        fig.tight_layout(pad=1)
        fig.show()



        # plot skree plot for explained variance
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(range(1, 1 + n_comp), pca.explained_variance_ratio_)
        fig.tight_layout(pad=1)
        fig.show()

        print(pca.explained_variance_ratio_)

        print(pca.singular_values_)

    @staticmethod
    def pca_responses_all_targets():
        results = PhotostimResponsesSLMtargetsResults.load()
        responses = results.baseline_adata.X
        z_scored_responses = stats.zscore(responses, axis=1)
        exps = results.baseline_adata.obs['exp']

        mean_responses = np.nanmean(responses, axis=1)
        variation_responses = variation(responses, axis=1)

        nanx = np.sum(np.isnan(z_scored_responses), axis=1)

        n_comp = 10
        pca = PCA(n_components=n_comp)
        pca_responses = pca.fit_transform(z_scored_responses.T)

        # clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(z_scored_responses)
        kmeans.labels_

        # plot PCA transformed responses
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, color='green', edgecolor='yellowgreen', linewidth=3)

        # sc = ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, c=kmeans.labels_, cmap=plt.cm.get_cmap('viridis', n_clusters))
        # fig.colorbar(sc, ticks=range(1), label ='kmeans clusters')

        # for exp in np.unique(exps):
        #     targets = np.where(exps == exp)
        #     sc = ax.scatter(pca_responses[targets, 0], pca_responses[targets, 1], s=50, label=exp)
        # ax.legend()

        # sc = ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, c=variation_responses, cmap=plt.cm.get_cmap('viridis', 10), vmin=0, vmax=3)
        # fig.colorbar(sc, ticks=range(10), label ='variation responses')

        # sc = ax.scatter(pca_responses[:, 0], pca_responses[:, 1], s=50, c=mean_responses, cmap='viridis', vmin= 0, vmax=100)
        # fig.colorbar(sc, label='mean responses')

        fig.tight_layout(pad=1)
        fig.show()



        # plot skree plot for explained variance
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(range(1, 1 + n_comp), pca.explained_variance_ratio_)
        fig.tight_layout(pad=1)
        fig.show()

        print(pca.explained_variance_ratio_)

        print(pca.singular_values_)

    # 2) calculating mean variability across targets:
    def calculate_variability(self, stims):
        """calculate coefficient of variation of photostim repsonses across trials for each inidividual target."""

        assert hasattr(self, 'adata'), 'cannot find .adata'
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        print('calculating coefficient of variation of photostim responses.')

        # cv = variation(self.adata.X[:, stims], axis=1)

        cv = np.nanstd(self.adata.X[:, stims] , ddof=1, axis=1) / np.nanmean(self.adata.X[:, stims] , axis=1)

        # cv = []
        # for target in self.adata.obs.index:
        #     target = int(target)
        #     cv_ = variation(self.adata.X[target, stims] / 100)  # dFF (not % dFF)
        #     # std_var = np.var(self.adata.X[target, stims] / 100, ddof=1)  # dFF (not % dFF)
        #     cv.append(cv_)

        return cv

    @staticmethod
    def plot__variability(rerun=False, fig_save_name='baseline-interictal_variability_of_photostim_responses.svg', **kwargs):
    # 1) plotting mean photostim response magnitude across experiments and experimental groups
        """create plot of mean photostim responses' COEFFICIENT OF VARIATION for all exp groups"""

        if rerun or not hasattr(results, 'variance_photostimresponse'):
            results.variance_photostimresponse = {}

            @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                            allow_rerun=1)
            def pre4apexps_collect_photostim_variation(**kwargs):
                expobj: alloptical = kwargs['expobj']
                if 'pre' in expobj.exptype:
                    # all stims
                    photostim_responses_cv = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims='all')
                    median_cv_baseline = np.median(photostim_responses_cv)
                    print(f'median baseline CV:', median_cv_baseline)

                    return np.abs(median_cv_baseline)

            photostim_variation_baseline = pre4apexps_collect_photostim_variation()
            results.variance_photostimresponse['baseline'] = photostim_variation_baseline

            @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False,
                                            allow_rerun=1)
            def post4apexps_collect_photostim_variation(**kwargs):
                expobj: Post4ap = kwargs['expobj']
                if 'post' in expobj.exptype:
                    # interictal stims
                    photostim_responses_cv_interictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_outsz)
                    median_cv_interictal = np.median(photostim_responses_cv_interictal)
                    print(f'median interictal CV:', median_cv_interictal)

                    # ictal stims
                    photostim_responses_cv_ictal = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims=expobj.stim_idx_insz)
                    median_cv_ictal = np.median(photostim_responses_cv_ictal)
                    print(f'median ictal CV:', median_cv_ictal)

                    return np.abs(median_cv_interictal), np.abs(median_cv_ictal)

            func_collector = post4apexps_collect_photostim_variation()

            # if len(func_collector) > 0:
            photostim_variation_interictal, photostim_variation_ictal = np.asarray(func_collector)[:, 0], np.asarray(func_collector)[:, 1]

            results.variance_photostimresponse['interictal'] = photostim_variation_interictal
            results.variance_photostimresponse['ictal'] = photostim_variation_ictal

            results.save_results()

        # fig, ax = pplot.plot_bar_with_points(
        #     data=[results.variance_photostimresponse['baseline'], results.variance_photostimresponse['interictal']],
        #     x_tick_labels=['Baseline', 'Interictal'], bar=True, colors=['gray', 'green'], alpha=1, show=False,
        #     expand_size_x=0.4, title='Average CV', y_label='Coefficient of Variation (CV)', ylims=[0, 50], shrink_text=0.9,
        #     **kwargs)
        # fig.tight_layout(pad=1)
        # fig.show()
        # Utils.save_figure(fig, save_path_suffix=f"{fig_save_name}") if fig_save_name else None
        plot_settings()

        res = mannwhitneyu(results.variance_photostimresponse['baseline'], results.variance_photostimresponse['interictal'])
        print(res)

        fig, ax = pplot.plot_bar_with_points(
            data=[results.variance_photostimresponse['baseline'], results.variance_photostimresponse['interictal']],
            x_tick_labels=['Baseline', 'Interictal'], bar=True, colors=['gray', 'green'], alpha=1, show=False,
            expand_size_x=0.4, title='Average CV', y_label='Coefficient of Variation (CV)', shrink_text=0.9,
            s=40, bar_alpha=0.2, ylims=[0, 7.5], **kwargs)
        ax.text(x=ax.get_xlim()[0] + 0.2,y=ax.get_ylim()[-1], s=f"MWU p-val: {res[1]:0.1e}", fontsize='xx-small')
        fig.tight_layout(pad=1)
        fig.show()
        Utils.save_figure(fig, save_path_suffix=f"{fig_save_name}") if fig_save_name else None

    # return photostim_stdvars_baseline, photostim_stdvars_interictal, photostim_stdvars_ictal

    @staticmethod
    def plot__schematic_variability_measurement():
        """plot for schematic of calculating variability"""
        amplitudes = np.random.rand(50)

        fig, ax = plt.subplots(figsize=[2, 2], dpi=300)
        total_x = []
        total_y = []
        ax.axvspan(-1.6, -0.75, alpha=0.35, color='hotpink')
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
        fig.tight_layout(pad=0.2)
        fig.show()

    # 2.2) plot mean magnitude of photostim responses vs. variability of photostim responses
    @staticmethod
    def plot__mean_response_vs_variability(rerun=False):
        """create plot of mean photostim responses vs. variability for all three exp groups"""

        fig, ax = plt.subplots(ncols=3)

        if rerun or not hasattr(results, 'meanresponses_vs_variance'):
            results.meanresponses_vs_variance = {}

            @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                            allow_rerun=1)
            def pre4apexps_collect_photostim_stdvars(**kwargs):
                expobj: alloptical = kwargs['expobj']
                assert 'pre' in expobj.exptype
                # all stims
                photostim_stdvars = expobj.PhotostimAnalysisSlmTargets.calculate_variability(stims='all')
                return photostim_stdvars


            @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False,
                                            allow_rerun=1)
            def pre4apexps_collect_photostim_responses(**kwargs):
                expobj: alloptical = kwargs['expobj']
                assert 'pre' in expobj.exptype
                # all stims
                mean_photostim_responses = expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgstims(
                    stims='all')
                return mean_photostim_responses


            mean_photostim_responses_baseline = pre4apexps_collect_photostim_responses()
            photostim_stdvars_baseline = pre4apexps_collect_photostim_stdvars()

            results.meanresponses_vs_variance['baseline - mean responses'] = mean_photostim_responses_baseline
            results.meanresponses_vs_variance['baseline - var responses'] = photostim_stdvars_baseline

            @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False,
                                            allow_rerun=1)
            def post4apexps_collect_photostim_stdvars(**kwargs):
                expobj: Post4ap = kwargs['expobj']
                assert 'post' in expobj.exptype
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
                assert 'post' in expobj.exptype
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

            results.meanresponses_vs_variance['interictal - mean responses'] = mean_photostim_responses_interictal
            results.meanresponses_vs_variance['interictal - var responses'] = photostim_stdvars_interictal

            results.meanresponses_vs_variance['ictal - mean responses'] = mean_photostim_responses_ictal
            results.meanresponses_vs_variance['ictal - var responses'] = photostim_stdvars_ictal

            results.save_results()


        # make plot
        pplot.make_general_scatter([pj.flattenOnce(results.meanresponses_vs_variance['baseline - mean responses'])],
                                   [pj.flattenOnce(results.meanresponses_vs_variance['baseline - var responses'])],
                                   x_labels=['Avg. Photostim Response (% dFF)'], y_labels = ['Variance (dFF^2)'], ax_titles=['Targets: avg. response vs. variability - baseline'],
                                   facecolors=['gray'], lw=1.3, figsize=(4,4), xlim=[-30, 130], ylim=[-0.1, 3.2], alpha=0.1)



        pplot.make_general_scatter([pj.flattenOnce(results.meanresponses_vs_variance['interictal - mean responses'])],
                                   [pj.flattenOnce(results.meanresponses_vs_variance['interictal - var responses'])],
                                   x_labels=['Avg. Photostim Response (% dFF)'], y_labels = ['Variance (dFF^2)'], ax_titles=['Targets: avg. response vs. variability - interictal'],
                                   facecolors=['forestgreen'], lw=1.3, figsize=(4,4), xlim=[-30, 130], ylim=[-0.1, 3.2], alpha=0.1)




        pplot.make_general_scatter([pj.flattenOnce(results.meanresponses_vs_variance['interictal - mean responses']), pj.flattenOnce(results.meanresponses_vs_variance['baseline - mean responses'])],
                                   [pj.flattenOnce(results.meanresponses_vs_variance['interictal - var responses']), pj.flattenOnce(results.meanresponses_vs_variance['baseline - var responses'])],
                                   x_labels=['Avg. Photostim Response (% dFF)'], y_labels = ['Variance (% dFF)^2'], ax_titles=['Targets: avg. response vs. variability - interictal'],
                                   facecolors=['forestgreen', 'royalblue'], lw=1.3, figsize=(4,4), xlim=[-30, 130], ylim=[-10, 32000], alpha=0.1)




        # pplot.make_general_scatter([pj.flattenOnce(results.meanresponses_vs_variance['ictal - mean responses'])],
        #                            [pj.flattenOnce(results.meanresponses_vs_variance['ictal - var responses'])],
        #                            x_labels=['Avg. Photostim Response (% dFF)'],
        #                            y_labels = ['Avg. standard dev. (% dFF)'], ax_titles=['Targets: avg. response vs. variability - ictal'],
        #                            facecolors=['none'], edgecolors=['mediumorchid'], lw=1.3, figsize=(4,4))

    # 3) interictal responses split by precital, very interictal, post ictal
    @staticmethod
    def collect__interictal_responses_split(rerun=0, RESULTS = results):
        data_label = 'z scored to baseline'

        @Utils.run_for_loop_across_exps(run_post4ap_trials=True, allow_rerun=1)
        def collect_avg_photostim_response_preictal(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            sz_onset_times = expobj.seizure_lfp_onsets
            preictal_stims = []
            for sz_onset in sz_onset_times:
                if sz_onset > 0:
                    for stim in expobj.stims_out_sz:
                        if (sz_onset - int(30 * expobj.fps)) < stim < sz_onset:
                            preictal_stims.append(expobj.stim_start_frames.index(stim))

            # print(preictal_stims)

            # take mean of response across preictal stims
            dff_responses = expobj.PhotostimResponsesSLMTargets.adata.X
            z_scored_responses = expobj.PhotostimResponsesSLMTargets.adata.layers[
                'dFF (zscored)']  # z scored to baseline responses

            if data_label == 'z scored to baseline':
                data_ = z_scored_responses
            else:
                data_ = dff_responses

            if len(expobj.preictal_stim_idx) > 0:
                avg_response = np.mean(data_[:, expobj.preictal_stim_idx], axis=1)
                return np.mean(avg_response)

        @Utils.run_for_loop_across_exps(run_post4ap_trials=True, allow_rerun=1)
        def collect_avg_photostim_response_postictal(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            # sz_offset_times = expobj.seizure_lfp_offsets
            # postictal_stims = []
            # for sz_offset in sz_offset_times:
            #     if sz_offset < expobj.n_frames:
            #         for stim in expobj.stims_out_sz:
            #             if (sz_offset - int(30 * expobj.fps)) < stim < sz_offset:
            #                 postictal_stims.append(expobj.stim_start_frames.index(stim))

            # take mean of response across preictal stims
            dff_responses = expobj.PhotostimResponsesSLMTargets.adata.X
            z_scored_responses = expobj.PhotostimResponsesSLMTargets.adata.layers[
                'dFF (zscored)']  # z scored to baseline responses

            if data_label == 'z scored to baseline':
                data_ = z_scored_responses
            else:
                data_ = dff_responses

            if len(expobj.postictal_stim_idx) > 0:
                avg_response = np.mean(data_[:, expobj.postictal_stim_idx], axis=1)
                return np.mean(avg_response)
            else:
                print(f'****** WARNING: no postictal stims for {expobj.t_series_name} ****** ')

        @Utils.run_for_loop_across_exps(run_post4ap_trials=True, allow_rerun=1)
        def collect_avg_photostim_response_very_interictal(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            # take mean of response across preictal stims
            dff_responses = expobj.PhotostimResponsesSLMTargets.adata.X
            z_scored_responses = expobj.PhotostimResponsesSLMTargets.adata.layers[
                'dFF (zscored)']  # z scored to baseline responses

            if data_label == 'z scored to baseline':
                data_ = z_scored_responses
            else:
                data_ = dff_responses

            if len(expobj.veryinterictal_stim_idx) > 0:
                avg_response = np.mean(data_[:, expobj.veryinterictal_stim_idx], axis=1)
                return np.mean(avg_response)
            else:
                print(f'****** WARNING: no postictal stims for {expobj.t_series_name} ****** ')

        if not hasattr(RESULTS, 'interictal_responses') or rerun:
            RESULTS.interictal_responses = {}
            RESULTS.interictal_responses['data_label'] = data_label
            RESULTS.interictal_responses['preictal_responses'] = collect_avg_photostim_response_preictal()
            RESULTS.interictal_responses['postictal_responses'] = collect_avg_photostim_response_postictal()
            RESULTS.interictal_responses['very_interictal_responses'] = collect_avg_photostim_response_very_interictal()
            RESULTS.save_results()




# 1.1) plot peri-photostim avg traces across all targets for all experiment trials
def plot_peristim_avg_photostims():
    plot = True

    for trial in pj.flattenOnce(AllOpticalExpsToAnalyze.pre_4ap_trials):
        if trial in PhotostimResponsesQuantificationSLMtargets.TEST_TRIALS:
            plot = True
        else: plot = False
        if plot:
            from _utils_.io import import_expobj
            expobj: alloptical = import_expobj(exp_prep=trial)
            trace_snippets_avg = np.mean(expobj.SLMTargets_tracedFF_stims_dff, axis=1)
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
            trace_snippets_avg = np.mean(expobj.SLMTargets_tracedFF_stims_dff, axis=1)
            print(trace_snippets_avg.shape[1], '\n')
            plot_periphotostim_avg(arr=trace_snippets_avg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                          title=f'{expobj.t_series_name} - post4ap', expobj=expobj,
                                          x_label='Time (secs)', y_label='dFF response', fig=fig, ax=axs[1])
            fig.show()

# 1.1.1) plot peri-fakestim avg traces across all targets for all experiment trials
def plot_peristim_avg_fakestims():
    plot = True

    for trial in pj.flattenOnce(AllOpticalExpsToAnalyze.pre_4ap_trials):
        # if trial in PhotostimResponsesQuantificationSLMtargets.TEST_TRIALS:
        #     plot = True
        # else:
        #     plot = False

        if plot:
            from _utils_.io import import_expobj
            expobj: alloptical = import_expobj(exp_prep=trial)
            trace_snippets_avg = np.mean(expobj.fake_SLMTargets_tracedFF_stims_dff, axis=1)
            print(trace_snippets_avg.shape[1])
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[4, 8])
            # aoplot.plot_periphotostim_avg2(dataset=trace_snippets_avg, fps=expobj.fps, pre_stim_sec=expobj.PhotostimAnalysisSlmTargets._pre_stim_sec, title=f'{expobj.t_series_name}')
            from _utils_.alloptical_plotting import plot_periphotostim_avg
            plot_periphotostim_avg(arr=trace_snippets_avg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                          title=f'{expobj.t_series_name} - pre4ap', expobj=expobj,
                                          x_label='Time (secs)', y_label='dFF response', fig=fig, ax=axs[0],
                                          show=False)

            fig.show()

    for trial in pj.flattenOnce(AllOpticalExpsToAnalyze.post_4ap_trials):
        # if trial in PhotostimResponsesQuantificationSLMtargets.TEST_TRIALS:
        #     plot = True
        # else:
        #     plot = False

        if plot:
            from _utils_.io import import_expobj
            expobj: Post4ap = import_expobj(exp_prep=trial)
            trace_snippets_avg = np.mean(expobj.fake_SLMTargets_tracedFF_stims_dff[:, expobj.fake_stim_idx_outsz], axis=1)
            print(trace_snippets_avg.shape[1])
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[4, 8])
            # aoplot.plot_periphotostim_avg2(dataset=trace_snippets_avg, fps=expobj.fps, pre_stim_sec=expobj.PhotostimAnalysisSlmTargets._pre_stim_sec, title=f'{expobj.t_series_name}')
            from _utils_.alloptical_plotting import plot_periphotostim_avg
            plot_periphotostim_avg(arr=trace_snippets_avg, pre_stim_sec=1.0, post_stim_sec=3.0,
                                          title=f'{expobj.t_series_name} - interictal', expobj=expobj,
                                          x_label='Time (secs)', y_label='dFF response', fig=fig, ax=axs[0],
                                          show=False)

            fig.show()



# 1.2) plot photostim avg of all targets from each experiment
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

    save_path_full = f'{SAVE_FIG}/alloptical_avg_photoresponses_allexps.svg'
    os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
    fig.savefig(save_path_full)



if __name__ == '__main__':
    main = PhotostimAnalysisSlmTargets
    main.run__init()
    # main.run__add_fps()

    # RUNNING PLOTS:
    # main.plot_postage_stamps_photostim_traces()
    # main.plot_variability_photostim_traces_by_targets()

    # main.correlation_matrix_responses()
    # main.pca_responses()


    main.collect_all_targets_all_exps()
    main.correlation_matrix_all_targets()
    main.pca_responses_all_targets()
    # main.plot__variability()
    # main.plot__mean_response_vs_variability()
    # plot__avg_photostim_dff_allexps()



# expobj = import_expobj(prep='RL108', trial='t-009')



# %%


