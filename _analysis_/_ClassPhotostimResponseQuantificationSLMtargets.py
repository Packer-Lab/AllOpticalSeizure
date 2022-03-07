from dataclasses import dataclass, field
from typing import Union, List, Dict

import numpy as np
import os
import pandas as pd

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
import funcsforprajay.funcs as pj

from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# %% COLLECT AND PLOT PHOTOSTIM RESPONSES MAGNITUDES

class PhotostimResponsesQuantificationSLMtargets(Quantification):
    # save_path = SAVE_LOC + 'PhotostimResponsesQuantificationSLMtargets.pkl'
    valid_zscore_processing_types = ['dFF (zscored)', 'dFF (zscored) (interictal)']
    valid_photostim_response_processing_types = ['dF/stdF', 'dF/prestimF', 'delta(trace_dFF)']

    def __init__(self, expobj: alloptical):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimResponsesSLMTargets MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"PhotostimResponsesSLMTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # %% 1)
    def collect_photostim_responses_exp(self, expobj: Union[alloptical, Post4ap]):
        """
        runs calculations of photostim responses, calculating reliability of photostim of slm targets,
        saving success stim locations, and saving stim response magnitudes as pandas dataframe.
        - of various methods -

        :param expobj: experiment trial object

        """

        # PRIMARY

        # dF/stdF
        self.StimSuccessRate_SLMtargets_dfstdf, self.hits_SLMtargets_dfstdf, self.responses_SLMtargets_dfstdf, self.traces_SLMtargets_successes_dfstdf = \
            expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                               stims_to_use=expobj.stim_start_frames)
        # dF/prestimF
        self.StimSuccessRate_SLMtargets_dfprestimf, self.hits_SLMtargets_dfprestimf, self.responses_SLMtargets_dfprestimf, self.traces_SLMtargets_successes_dfprestimf = \
            expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                               stims_to_use=expobj.stim_start_frames)
        # trace dFF
        self.StimSuccessRate_SLMtargets_tracedFF, self.hits_SLMtargets_tracedFF, self.responses_SLMtargets_tracedFF, self.traces_SLMtargets_tracedFF_successes = \
            expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10,
                                               stims_to_use=expobj.stim_start_frames)

        f, ax = pplot.make_general_scatter(x_list=[np.random.random(self.responses_SLMtargets_tracedFF.shape[0])],
                                           y_data=[np.mean(self.responses_SLMtargets_tracedFF, axis=1)],
                                           ax_titles=[f"{expobj.t_series_name}"], show=False,
                                           y_label='delta(trace_dff)', figsize=[2, 4],
                                           x_lim=[-1, 2], y_lim=[-50, 100])
        ax.set_xticks([0.5])
        ax.set_xticklabels(['targets'])
        f.show()

        # SECONDARY - SPLIT DOWN BY STIMS IN AND OUT OF SZ FOR POST4AP TRIALS
        ### STIMS OUT OF SEIZURE
        if 'post' in expobj.exptype:
            if expobj.stims_out_sz:
                stims_outsz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
                if stims_outsz_idx:
                    print('|- calculating stim responses (outsz) - %s stims [2.2.1]' % len(stims_outsz_idx))
                    # dF/stdF
                    self.StimSuccessRate_SLMtargets_dfstdf_outsz, self.hits_SLMtargets_dfstdf_outsz, self.responses_SLMtargets_dfstdf_outsz, self.traces_SLMtargets_successes_dfstdf_outsz = \
                        expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                                           stims_to_use=expobj.stims_out_sz)
                    # dF/prestimF
                    self.StimSuccessRate_SLMtargets_dfprestimf_outsz, self.hits_SLMtargets_dfprestimf_outsz, self.responses_SLMtargets_dfprestimf_outsz, self.traces_SLMtargets_successes_dfprestimf_outsz = \
                        expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                                           stims_to_use=expobj.stims_out_sz)
                    # trace dFF
                    self.StimSuccessRate_SLMtargets_tracedFF_outsz, self.hits_SLMtargets_tracedFF_outsz, self.responses_SLMtargets_tracedFF_outsz, self.traces_SLMtargets_tracedFF_successes_outsz = \
                        expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10,
                                                           stims_to_use=expobj.stims_out_sz)

            ### STIMS IN SEIZURE
            if expobj.stims_in_sz:
                if hasattr(expobj, 'slmtargets_szboundary_stim'):
                    stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    if stims_insz_idx:
                        print('|- calculating stim responses (insz) - %s stims [2.3.1]' % len(stims_insz_idx))
                        # dF/stdF
                        self.StimSuccessRate_SLMtargets_dfstdf_insz, self.hits_SLMtargets_dfstdf_insz, self.responses_SLMtargets_dfstdf_insz, self.traces_SLMtargets_successes_dfstdf_insz = \
                            expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
                                                               stims_to_use=expobj.stims_in_sz)
                        # dF/prestimF
                        self.StimSuccessRate_SLMtargets_dfprestimf_insz, self.hits_SLMtargets_dfprestimf_insz, self.responses_SLMtargets_dfprestimf_insz, self.traces_SLMtargets_successes_dfprestimf_insz = \
                            expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
                                                               stims_to_use=expobj.stims_in_sz)
                        # trace dFF
                        self.StimSuccessRate_SLMtargets_tracedFF_insz, self.hits_SLMtargets_tracedFF_insz, self.responses_SLMtargets_tracedFF_insz, self.traces_SLMtargets_tracedFF_successes_insz = \
                            expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10,
                                                               stims_to_use=expobj.stims_in_sz)


                    else:
                        print(f'******* No stims in sz for: {expobj.t_series_name}', ' [*2.3] ')


                else:
                    print(
                        f'******* No slmtargets_szboundary_stim (sz boundary classification not done) for: {expobj.t_series_name}',
                        ' [*2.3] ')

    # %% 2) create anndata SLM targets to store photostim responses for each experiment
    def create_anndata_SLMtargets(self, expobj: Union[alloptical, Post4ap]):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial.

        """

        if not (hasattr(self, 'responses_SLMtargets_tracedFF') or hasattr(self,
                                                                          'responses_SLMtargets_dfprestimf') or hasattr(
            self, 'responses_SLMtargets_dfstdf')):
            raise Warning(
                'did not create anndata. anndata creation only available if experiments were processed with suite2p and .paq file(s) provided for temporal synchronization')
        else:
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
                var_meta.loc['im_time_secs', fr_idx] = stim_frame * expobj.fps

            # SET PRIMARY DATA
            _data_type = 'SLM Targets photostim responses delta(tracedFF)'  # primary data label
            self.responses_SLMtargets_tracedFF.columns = range(len(expobj.stim_start_frames))
            photostim_responses = self.responses_SLMtargets_tracedFF

            # BUILD LAYERS TO ADD TO anndata OBJECT
            self.responses_SLMtargets_dfstdf.columns = range(len(expobj.stim_start_frames))
            self.responses_SLMtargets_dfprestimf.columns = range(len(expobj.stim_start_frames))
            layers = {'SLM Targets photostim responses (dF/stdF)': self.responses_SLMtargets_dfstdf,
                      'SLM Targets photostim responses (dF/prestimF)': self.responses_SLMtargets_dfprestimf
                      }

            print(f"\t\----- CREATING annotated data object using AnnData:")
            # create anndata object
            photostim_responses_adata = AnnotatedData2(X=photostim_responses, obs=obs_meta, var=var_meta.T, obsm=obs_m,
                                                       layers=layers, data_label=_data_type)

            print(f"Created: {photostim_responses_adata}")
            self.adata = photostim_responses_adata

    # 2.1) modify anndata
    def add_stim_group_anndata(self, expobj: Union[alloptical, Post4ap]):
        new_var = pd.Series(name='stim_group', index=self.adata.var.index, dtype='str')

        for fr_idx in self.adata.var.index:
            if 'pre' in self.expobj_exptype:
                new_var[fr_idx] = 'baseline'
            elif 'post' in self.expobj_exptype:
                new_var[fr_idx] = 'interictal' if expobj.stim_start_frames[
                                                      int(fr_idx)] in expobj.stims_out_sz else 'ictal'

        self.adata.add_variable(var_name=str(new_var.name), values=list(new_var))

    # %% 3) PLOTTING MEAN PHOTOSTIM RESPONSE AMPLITUDES

    def collect_photostim_responses_magnitude(self, stims: Union[slice, str, list] = None):
        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        mean_photostim_responses = []
        for target in self.adata.obs.index:
            target = int(target)
            mean_photostim_response = np.mean(self.adata.X[target, stims])
            mean_photostim_responses.append(mean_photostim_response)
        return mean_photostim_responses

    def plot_photostim_responses_magnitude(self, expobj: alloptical, stims: Union[slice, str, list] = None):
        """quick plot of photostim responses of expobj's targets across all stims"""
        mean_photostim_responses = self.collect_photostim_responses_magnitude(stims)
        x_scatter = [float(np.random.rand(1) * 1)] * len(mean_photostim_responses)
        pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses],
                                   ax_titles=[expobj.t_series_name],
                                   figsize=[2, 4], y_label='delta(trace_dFF)')
        # pplot.plot_bar_with_points(data=[mean_photostim_responses], bar = False, title=expobj.t_series_name)

    # 3.1) Plotting mean photostim response amplitude across experiments
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=True, set_cache=False)
    def allexps_plot_photostim_responses_magnitude(**kwargs):
        expobj: alloptical = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.plot_photostim_responses_magnitude(expobj=expobj, stims='all')

    # %% 4) Zscoring of photostimulation responses
    def z_score_photostim_responses(self):
        """
        z scoring of photostimulation response across all stims for each target.

        """
        print(f"\t\- zscoring photostim responses across all stims in expobj trial")

        df = pd.DataFrame(index=self.adata.obs.index, columns=self.adata.var.index)

        _slice_ = [int(idx) for idx in self.adata.var.index]  # (slice using all stims)

        for target in self.adata.obs.index:
            # z-scoring of SLM targets responses:
            _mean_ = self.adata.X[int(target), _slice_].mean()
            _std_ = self.adata.X[int(target), _slice_].std(ddof=1)

            __responses = self.adata.X[int(target), :]
            z_scored_stim_response = (__responses - _mean_) / _std_
            df.loc[target, :] = z_scored_stim_response

        # add zscored data to anndata storage
        self.adata.add_layer(layer_name='dFF (zscored)', data=np.asarray(df))

    def z_score_photostim_responses_interictal(self):
        """
        z scoring of photostimulation response across all interictal stims for each target.

        :param expobj:
        :param response_type: either 'dFF (z scored)' or 'dFF (z scored) (interictal)'
        """
        print(f"\t\- zscoring photostim responses (to interictal stims) across all stims in expobj trial")

        df = pd.DataFrame(index=self.adata.obs.index, columns=self.adata.var.index)

        _slice_ = [idx for idx, val in enumerate(self.adata.var['stim_group']) if
                   val == 'interictal']  # (slice using all interictal stims)

        if 'post' not in self.expobj_exptype:
            print(f'\t ** not running interictal zscoring on non-Post4ap expobj **')
        else:
            for target in self.adata.obs.index:
                # z-scoring of SLM targets responses:
                _mean_ = self.adata.X[int(target), _slice_].mean()
                _std_ = self.adata.X[int(target), _slice_].std(ddof=1)

                __responses = self.adata.X[int(target), :]
                z_scored_stim_response = (__responses - _mean_) / _std_
                df.loc[target, :] = z_scored_stim_response

        # add zscored data to anndata storage
        self.adata.add_layer(layer_name='dFF (zscored) (interictal)', data=np.asarray(df))

    # 4.1) plot z scored photostim responses
    def collect_photostim_responses_magnitude_zscored(self, zscore_type: str = 'dFF (zscored)',
                                                      stims: Union[slice, str, list] = None):
        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        if 'pre' in self.expobj_exptype:
            zscore_type = 'dFF (zscored)'  # force zscore_type to always be this for pre4ap experiments

        mean_zscores_stims = np.mean(self.adata.layers[zscore_type][:, stims], axis=0)

        return mean_zscores_stims

    def plot_photostim_responses_magnitude_zscored(self, zscore_type: str = 'dFF (zscored)',
                                                   stims: Union[slice, str, list] = None):
        """quick plot of photostim responses of expobj's targets across all stims"""
        mean_photostim_responses_zscored = self.collect_photostim_responses_magnitude_zscored(zscore_type=zscore_type,
                                                                                              stims=stims)
        x_scatter = [float(np.random.rand(1) * 1)] * len(mean_photostim_responses_zscored)
        fig, ax = pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses_zscored],
                                             ax_titles=[self.expobj_id], show=False, figsize=[2, 4],
                                             y_label=zscore_type)
        ax.set_xticks([])
        fig.show()
        # pplot.plot_bar_with_points(data=[mean_photostim_responses], bar = False, title=expobj.t_series_name)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, set_cache=0)
    def allexps_plot_photostim_responses_magnitude_zscored(**kwargs):
        expobj: alloptical = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.plot_photostim_responses_magnitude_zscored(zscore_type='dFF (zscored)',
                                                                                       stims='all')


# %%
@dataclass
class PhotostimResponsesSLMtargetsResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesSLMtargets.pkl'
    mean_photostim_responses_baseline: List[float] = field(default_factory=lambda: [-1])
    mean_photostim_responses_interictal: List[float] = field(default_factory=lambda: [-1])
    mean_photostim_responses_ictal: List[float] = field(default_factory=lambda: [-1])

    mean_photostim_responses_baseline_zscored: List[float] = field(default_factory=lambda: [-1])
    mean_photostim_responses_interictal_zscored: List[float] = field(default_factory=lambda: [-1])
    mean_photostim_responses_ictal_zscored: List[float] = field(default_factory=lambda: [-1])

    pre_stim_FOV_flu: Dict[
                      str: List] = None  # averages from pre-stim Flu value for each stim frame for baseline, interictal and ictal groups

    @classmethod
    def load(cls):
        return pj.load_pkl(cls.SAVE_PATH)


if not os.path.exists(PhotostimResponsesSLMtargetsResults.SAVE_PATH):
    RESULTS = PhotostimResponsesSLMtargetsResults()
    RESULTS.save_results()

RESULTS = PhotostimResponsesSLMtargetsResults.load()

# %% CODE DEVELOPMENT ZONE

"""
TODO:
1. Measuring photostim responses in relation to pre-stim mean FOV Flu - main priority right now


2. measuring photostim responses of targets (suite2p rois) vs. pre-stim surrounding neuropil signal -- not immediately setup yet to do analysis involving suite2p
    - need to ensure that you have the adata structure for slm targets that are also suite2p ROIs
    - another approach that sidesteps suite2p is just directly grabbing a torus of area around the SLM target

"""

if __name__ == '__main__':
    # 5) Measuring photostim responses in relation to pre-stim mean FOV Flu - main priority right now

    def collect__prestim_FOV_Flu():
        """
        two act function that collects pre-stim FOV Flu value for each stim frame.

        1) collect pre-stim FOV Flu value for each stim frame. Add these as a var to the expobj.PhotostimResponsesSLMTargets.adata table
            - length of the pre-stim == expobj.pre_stim

        2) collect average prestim FOV values values across various stim group types.

        """

        import alloptical_utils_pj as aoutils
        expobj: aoutils.Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

        # PART 1)   ####################################################################################################

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False)
        def __collect_prestim_FOV_Flu_allstims(**kwargs):
            expobj: aoutils.alloptical = kwargs['expobj']
            pre_stim_FOV_flu = []
            for stim in expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame:
                sli_ce = np.s_[stim - expobj.pre_stim: stim]
                _pre_stim_FOV_flu = expobj.meanRawFluTrace[sli_ce]
                pre_stim_FOV_flu.append(np.round(np.mean(_pre_stim_FOV_flu), 3))

            expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='pre_stim_FOV_Flu', values=pre_stim_FOV_flu)
            expobj.save()

        __collect_prestim_FOV_Flu_allstims()

        # PART 2)   ####################################################################################################
        # TODO edit code to collect the prestimF FOV flu values from the anndata table to have it all centralized (rather than recollecting the prestim FOV flus for each stim

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False)
        def __collect_prestim_FOV_Flu_pre4ap(**kwargs):
            expobj: aoutils.alloptical = kwargs['expobj']

            if 'pre' in expobj.exptype:
                # pre_stim_FOV_flu = []
                # for stim in expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame:
                #     sli_ce = np.s_[stim - expobj.pre_stim: stim]
                #     _pre_stim_FOV_flu = expobj.meanRawFluTrace[sli_ce]
                #     pre_stim_FOV_flu.append(np.round(np.mean(_pre_stim_FOV_flu), 3))
                #
                # baseline_pre_stim_FOV_flu = pre_stim_FOV_flu

                ###
                baseline_pre_stim_FOV_flu = expobj.PhotostimResponsesSLMTargets.adata.var.pre_stim_FOV_Flu

                return baseline_pre_stim_FOV_flu

        baseline_pre_stim_FOV_flu = __collect_prestim_FOV_Flu_pre4ap()

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=False)
        def __collect_prestim_FOV_Flu_post4ap(**kwargs):
            expobj: aoutils.Post4ap = kwargs['expobj']

            if 'post' in expobj.exptype:
                # collect interictal stims ########
                interictal_stims_idx = \
                    np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_group == 'interictal')[0]
                # pre_stim_FOV_flu_interic = []
                # for stim in expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame[interictal_stims_idx]:
                #     sli_ce = np.s_[stim - expobj.pre_stim: stim]
                #     _pre_stim_FOV_flu = expobj.meanRawFluTrace[sli_ce]
                #     pre_stim_FOV_flu_interic.append(np.round(np.mean(_pre_stim_FOV_flu), 3))
                #
                # interictal_pre_stim_FOV_flu = pre_stim_FOV_flu_interic

                ###
                interictal_pre_stim_FOV_flu = expobj.PhotostimResponsesSLMTargets.adata.var.pre_stim_FOV_Flu[interictal_stims_idx]

                # collect ictal stims ########
                ictal_stims_idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_group == 'ictal')[0]
                # pre_stim_FOV_flu_ic = []
                # for stim in expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame[ictal_stims_idx]:
                #     sli_ce = np.s_[stim - expobj.pre_stim: stim]
                #     _pre_stim_FOV_flu = expobj.meanRawFluTrace[sli_ce]
                #     pre_stim_FOV_flu_ic.append(np.round(np.mean(_pre_stim_FOV_flu), 3))
                #
                # ictal_pre_stim_FOV_flu = pre_stim_FOV_flu_ic

                ###
                ictal_pre_stim_FOV_flu = expobj.PhotostimResponsesSLMTargets.adata.var.pre_stim_FOV_Flu[ictal_stims_idx]


                return interictal_pre_stim_FOV_flu, ictal_pre_stim_FOV_flu

        func_collector = __collect_prestim_FOV_Flu_post4ap()

        assert len(func_collector) > 0, '__collect_prestim_FOV_Flu_post4ap didnot return any results.'

        interictal_pre_stim_FOV_flu, ictal_pre_stim_FOV_flu = np.asarray(func_collector)[:, 0], np.asarray(
            func_collector)[:, 1]

        # process returned data to make flat arrays
        pre_stim_FOV_flu_results = {'baseline': baseline_pre_stim_FOV_flu,
                                    'interictal': interictal_pre_stim_FOV_flu,
                                    'ictal': ictal_pre_stim_FOV_flu}

        # PART 2)   ####################################################################################################
        return pre_stim_FOV_flu_results


    RESULTS.pre_stim_FOV_flu = collect__prestim_FOV_Flu()
    RESULTS.save_results()


    def plot__prestim_FOV_Flu(**kwargs):
        """plot avg pre-stim Flu values across baseline, interictal, and ictal stims"""
        import alloptical_utils_pj as aoutils
        expobj: aoutils.Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

        RESULTS.pre_stim_FOV_flu
        pass


    def measure__photostim_responses_vs_prestim_FOV_flu(**kwargs):
        import alloptical_utils_pj as aoutils
        expobj: aoutils.Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

        # measure/plot avg target photostim responses in relation to pre-stim Flu value across baseline, interictal, and ictal stims
        for target in range(expobj.PhotostimResponsesSLMTargets.adata.n_obs):
            pass


    pass
