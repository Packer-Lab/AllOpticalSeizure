from typing import Union, List, Dict

import seaborn as sns
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
import funcsforprajay.funcs as pj

from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

# %% COLLECTING DATA, PROCESSING AND ANALYSIS FOR PHOTOSTIM RESPONSES MAGNITUDES OF SLM TARGETS

"""

[ ] create fake sham trials to use as artificial 'catch' trials 


"""


class PhotostimResponsesSLMtargetsResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesSLMtargets.pkl'

    def __init__(self):
        super().__init__()
        self.pre_stim_targets_annulus_vs_targets_responses_results = None
        self.mean_photostim_responses_baseline: List[float] = [-1]
        self.mean_photostim_responses_interictal: List[float] = [-1]
        self.mean_photostim_responses_ictal: List[float] = [-1]

        self.mean_photostim_responses_baseline_zscored: List[float] = [-1]
        self.mean_photostim_responses_interictal_zscored: List[float] = [-1]
        self.mean_photostim_responses_ictal_zscored: List[float] = [-1]

        self.pre_stim_FOV_flu = None  # averages from pre-stim Flu value for each stim frame for baseline, interictal and ictal groups
        self.baseline_pre_stim_targets_annulus = None
        self.interictal_pre_stim_targets_annulus = None
        self.ictal_pre_stim_targets_annulus = None
        self.expavg_pre_stim_targets_annulus_F = None
        self.expavg_pre_stim_targets_annulus_results_ictal = None

        self.variance_photostimresponse = {}

        self.interictal_responses = {'data_label': '',
            'very_interictal_responses': [],
            'preictal_responses': [],
                                     'postictal_responses': []}  #: responses during interictal split by preictal and postictal

        self.baseline_adata: AnnotatedData2 = None      #: baseline dFF repsonses all targets, all exps, all stims
        self.interictal_adata: AnnotatedData2 = None    #: interictal dFF repsonses all targets, all exps, all stims

REMAKE = False
if not os.path.exists(PhotostimResponsesSLMtargetsResults.SAVE_PATH) or REMAKE:
    RESULTS = PhotostimResponsesSLMtargetsResults()
    RESULTS.save_results()
else:
    RESULTS = PhotostimResponsesSLMtargetsResults.load()


class PhotostimResponsesQuantificationSLMtargets(Quantification):
    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationSLMtargets.pkl'
    valid_zscore_processing_types = ['dFF (zscored)', 'dFF (zscored) (interictal)']
    valid_photostim_response_processing_types = ['dF/stdF', 'dF/prestimF', 'delta(trace_dFF)']


    TEST_TRIALS = [
        'RL108 t-009'
    ]

    def __init__(self, expobj: alloptical):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimResponsesSLMTargets MODULE to expobj: {expobj.t_series_name}')
        self.fake_stim_frames = None  #: fake stim time frames - create fake sham stim frames - halfway in between each stim trial


    def __repr__(self):
        return f"PhotostimResponsesSLMTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0) IDENTIFY, GROUP, CLASSIFY, CREATE PHOTOSTIMULATION FRAMES

    # 0.1) SET FAKE-SHAM PHOTOSTIMULATION TRIALS
    @staticmethod
    def plot__fake_sham_photostim(expobj: Union[alloptical, Post4ap], plot=False):
        """set fake sham stim frames - halfway in between each stim trial"""
        fake_stims = expobj.fake_stim_start_frames
        if plot:
            # make plot to confirm timings line up:
            fig, ax = plt.subplots(figsize=[10, 2])
            ax.plot(range(expobj.n_frames), [0] * expobj.n_frames)
            for stim in expobj.stim_start_frames:
                ax.axvline(stim, color='red')
            for fake in fake_stims:
                ax.axvline(fake, color='gray')
            ax.set_xticks(np.arange(0, expobj.n_frames, 60 * expobj.fps))
            ax.set_xticklabels([int(time) for time in np.arange(0, expobj.n_frames / expobj.fps, 60)])
            ax.set_xlabel('Time (secs)')
            fig.tight_layout(pad=2)
            ax.set_yticks([])
            ax.set_title(f"{expobj.t_series_name}")
            fig.show()

        return fake_stims





    # 1) collect various photostimulation success rates, matrix of hits/failures, responses sizes and photostim stims trace snippets of SLM targets
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
                                           y_data=[np.mean(self.responses_SLMtargets_tracedFF, axis=1)], show=False,
                                           y_label='delta(trace_dff)', figsize=[2, 4], x_lim=[-1, 2], y_lim=[-50, 100])
        ax.set_xticks([0.5])
        ax.set_xticklabels(['targets'])
        ax.set_title(f"{expobj.t_series_name} - photostim avg responses", wrap = True)
        f.tight_layout(pad=2)
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
                        expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3, stims_to_use=expobj.stims_out_sz)

                    # dF/prestimF
                    self.StimSuccessRate_SLMtargets_dfprestimf_outsz, self.hits_SLMtargets_dfprestimf_outsz, self.responses_SLMtargets_dfprestimf_outsz, self.traces_SLMtargets_successes_dfprestimf_outsz = \
                        expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10, stims_to_use=expobj.stims_out_sz)
                    # trace dFF
                    self.StimSuccessRate_SLMtargets_tracedFF_outsz, self.hits_SLMtargets_tracedFF_outsz, self.responses_SLMtargets_tracedFF_outsz, self.traces_SLMtargets_tracedFF_successes_outsz = \
                        expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10, stims_to_use=expobj.stims_out_sz)

            ### STIMS IN SEIZURE
            if expobj.stims_in_sz:
                if hasattr(expobj, 'slmtargets_szboundary_stim'):
                    stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    if stims_insz_idx:
                        print('|- calculating stim responses (insz) - %s stims [2.3.1]' % len(stims_insz_idx))
                        # dF/stdF
                        self.StimSuccessRate_SLMtargets_dfstdf_insz, self.hits_SLMtargets_dfstdf_insz, self.responses_SLMtargets_dfstdf_insz, self.traces_SLMtargets_successes_dfstdf_insz = \
                            expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3, stims_to_use=expobj.stims_in_sz)
                        # dF/prestimF
                        self.StimSuccessRate_SLMtargets_dfprestimf_insz, self.hits_SLMtargets_dfprestimf_insz, self.responses_SLMtargets_dfprestimf_insz, self.traces_SLMtargets_successes_dfprestimf_insz = \
                            expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10, stims_to_use=expobj.stims_in_sz)
                        # trace dFF
                        self.StimSuccessRate_SLMtargets_tracedFF_insz, self.hits_SLMtargets_tracedFF_insz, self.responses_SLMtargets_tracedFF_insz, self.traces_SLMtargets_tracedFF_successes_insz = \
                            expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10, stims_to_use=expobj.stims_in_sz)


                    else:
                        print(f'******* No stims in sz for: {expobj.t_series_name}', ' [*2.3] ')


                else:
                    print(
                        f'******* No slmtargets_szboundary_stim (sz boundary classification not done) for: {expobj.t_series_name}',
                        ' [*2.3] ')


    def collect_fake_photostim_responses_exp(self, expobj: Union[alloptical, Post4ap]):
        """
        Uses fake photostim stim timing frames. JUST IMPLEMENTED FOR BASELINE STIMS FOR NOW.
        runs calculations of photostim responses, calculating reliability of photostim of slm targets,
        saving success stim locations, and saving stim response magnitudes as pandas dataframe.
        - of various methods -

        :param expobj: experiment trial object

        """
        # NOTE: ONLY IMPLEMENTED FOR TRACE_DFF PROCESSED TRACES SO FAR.

        # PRIMARY

        # # dF/stdF
        # self.fake_StimSuccessRate_SLMtargets_dfstdf, self.fake_hits_SLMtargets_dfstdf, self.fake_responses_SLMtargets_dfstdf, self.fake_traces_SLMtargets_successes_dfstdf = \
        #     expobj.get_SLMTarget_responses_dff(process='dF/stdF', threshold=0.3,
        #                                        stims_to_use='fake_stims')
        # # dF/prestimF
        # self.fake_StimSuccessRate_SLMtargets_dfprestimf, self.fake_hits_SLMtargets_dfprestimf, self.fake_responses_SLMtargets_dfprestimf, self.fake_traces_SLMtargets_successes_dfprestimf = \
        #     expobj.get_SLMTarget_responses_dff(process='dF/prestimF', threshold=10,
        #                                        stims_to_use='fake_stims')

        # trace dFF
        self.fake_StimSuccessRate_SLMtargets_tracedFF, self.fake_hits_SLMtargets_tracedFF, self.fake_responses_SLMtargets_tracedFF, self.fake_traces_SLMtargets_tracedFF_successes = \
            expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10,
                                               stims_to_use='fake_stims')

        f, ax = pplot.make_general_scatter(x_list=[np.random.random(self.fake_responses_SLMtargets_tracedFF.shape[0])],
                                           y_data=[np.mean(self.fake_responses_SLMtargets_tracedFF, axis=1)], show=False,
                                           y_label='delta(trace_dff)', figsize=[2, 4], x_lim=[-1, 2], y_lim=[-50, 100])
        ax.set_xticks([0.5])
        ax.set_xticklabels(['targets'])
        ax.set_title(f"{expobj.t_series_name} - fakestims avg responses", wrap = True)
        f.tight_layout(pad=2)
        f.show()




    # 2) create anndata SLM targets to store photostim responses for each experiment
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

    # 2.1) add to anndata - stim groups (baseline, interictal vs. ictal)
    def add_stim_group_anndata(self, expobj: Union[alloptical, Post4ap]):
        new_var = pd.Series(name='stim_group', index=self.adata.var.index, dtype='str')

        for fr_idx in self.adata.var.index:
            if 'pre' in self.expobj_exptype:
                new_var[fr_idx] = 'baseline'
            elif 'post' in self.expobj_exptype:
                new_var[fr_idx] = 'interictal' if expobj.stim_start_frames[
                                                      int(fr_idx)] in expobj.stims_out_sz else 'ictal'

        self.adata.add_variable(var_name=str(new_var.name), values=list(new_var))

    @property
    def interictal_stims_idx(self):
        """getter to index interictal stims from anndata object after stim groups are added as var to adata."""
        assert 'stim_group' in self.adata.var_keys()
        return np.where(self.adata.var.stim_group == 'interictal')[0]

    @property
    def ictal_stims_idx(self):
        assert 'stim_group' in self.adata.var_keys()
        return np.where(self.adata.var.stim_group == 'ictal')[0]

    # 2.2) add to anndata - hit trials
    def add_hit_trials_anndata(self):
        """
        hit trials: successful photostimulation trials across experiments.
        - threshold for successful photostimulation trials: >10% dFF response

        add success (1) or failure (0) at each target/stim matrix as a layer to anndata.
        """

        self.adata.add_layer(layer_name='hits/failures (trade_dff)', data=self.hits_SLMtargets_tracedFF)

    # 2.0.1) add to anndata - fake photostim responses as layer
    def add_fakestim_adata_layer(self):
        # assert 'pre' in self.expobj_exptype, 'fakestim responses currenty only available for pre4ap baseline trials'
        fakestim_responses = self.fake_responses_SLMtargets_tracedFF
        self.adata.add_layer(layer_name='fakestim_responses', data=fakestim_responses)


    # 3) PLOTTING MEAN PHOTOSTIM RESPONSE AMPLITUDES
    def collect_photostim_responses_magnitude_avgtargets(self, stims: Union[slice, str, list] = 'all',
                                                         targets: Union[slice, str, list] = 'all',
                                                         adata_layer: str = 'primary'):
        "collect avg photostim responses of targets overall individual stims - add to .adata.var"
        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = range(self.adata.n_vars)
        if not targets or targets == 'all': targets = slice(0, self.adata.n_obs)

        if adata_layer == 'primary':
            df = self.adata.X
        elif adata_layer in self.adata.layers:
            df = self.adata.layers[adata_layer]
        else:
            raise ValueError(f"`{adata_layer}` layer not found to collect photostim response magnitudes from.")

        # select stims to use to collect data from
        mean_photostim_responses = []
        for stim in stims:
            mean_photostim_response = np.mean(df[targets, stim])
            mean_photostim_responses.append(mean_photostim_response)

        self.adata.add_variable(var_name='avg targets photostim response', values=mean_photostim_responses)

        # return mean_photostim_responses

    def collect_fakestim_responses_magnitude_avgtargets(self, stims: Union[slice, str, list] = 'all',
                                                         targets: Union[slice, str, list] = 'all',
                                                         adata_layer: str = 'fakestim_responses'):
        "collect avg fakestim responses of targets overall individual stims - add to .adata.var"
        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = range(self.adata.n_vars)
        if not targets or targets == 'all': targets = slice(0, self.adata.n_obs)

        assert adata_layer in self.adata.layers, f"`{adata_layer}` layer not found to collect fakestim response magnitudes from."
        df = self.adata.layers[adata_layer]

        # select stims to use to collect data from
        mean_fakestim_responses = []
        for stim in stims:
            mean_fakestim_response = np.mean(df[targets, stim])
            mean_fakestim_responses.append(mean_fakestim_response)

        self.adata.add_variable(var_name='avg targets fakestim response', values=mean_fakestim_responses)

        # return mean_photostim_responses


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
    def run__collect_photostim_responses_magnitude_avgtargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.collect_photostim_responses_magnitude_avgtargets(stims='all', targets='all',
                                                                                             adata_layer='primary')

        if 'pre' in expobj.exptype:
            expobj.PhotostimResponsesSLMTargets.collect_fakestim_responses_magnitude_avgtargets(stims='all', targets='all', adata_layer='fakestim_responses')

        expobj.save()


    # 3.1) COLLECT PHOTOSTIM RESPONSES FOR TARGETS AVG ACROSS ALL STIMS
    def collect_photostim_responses_magnitude_avgstims(self, stims: Union[slice, str, list] = 'all',
                                                       adata_layer: str = 'primary'):
        """collect mean photostim response magnitudes over all stims specified for all targets.
        the type of photostim response magnitude collected is specified by adata_layer (where 'primary' just means the primary adata layer). check .adata.layers to see available options.
        """

        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        if adata_layer == 'primary':
            df = self.adata.X
        elif adata_layer in self.adata.layers:
            df = self.adata.layers[adata_layer]
        else:
            raise ValueError(f"`{adata_layer}` layer not found to collect photostim response magnitudes from.")

        mean_photostim_responses = []
        for target in self.adata.obs.index:
            target = int(target)
            mean_photostim_response = np.mean(df[target, stims])
            mean_photostim_responses.append(mean_photostim_response)
        return mean_photostim_responses
        
    def collect_fakestim_responses_magnitude_avgstims(self, stims: Union[slice, str, list] = 'all',
                                                       adata_layer: str = 'fakestim_responses'):
        """collect mean fakestim response magnitudes over all stims specified for all targets.
        the type of fakestim response magnitude collected is specified by adata_layer (where 'primary' just means the primary adata layer). check .adata.layers to see available options.
        """

        assert self.adata, print('cannot find .adata')
        if not stims or stims == 'all': stims = slice(0, self.adata.n_vars)

        assert adata_layer in self.adata.layers, f"`{adata_layer}` layer not found to collect photostim response magnitudes from."
        df = self.adata.layers[adata_layer]

        mean_fakestim_responses = []
        for target in self.adata.obs.index:
            target = int(target)
            mean_fakestim_response = np.mean(df[target, stims])
            mean_fakestim_responses.append(mean_fakestim_response)
        return mean_fakestim_responses


    # todo add plot of mean fakestim responses magnitude --> full_plot_mean_responses_magnitudes_zscored

    def plot_photostim_responses_magnitude(self, expobj: alloptical, stims: Union[slice, str, list] = None):
        """quick plot of photostim responses of expobj's targets across all stims"""
        mean_photostim_responses = self.collect_photostim_responses_magnitude_avgstims(stims)

        if 'pre' in expobj.exptype:
            # todo add plot of mean fakestim responses magnitude

            mean_fakestim_responses = self.collect_fakestim_responses_magnitude_avgstims(stims)
            x_scatter = [float(np.random.rand(1) * 1)] * len(mean_photostim_responses)
            pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_fakestim_responses],
                                       ax_titles=[f"{expobj.t_series_name} - fakestims"],
                                       figsize=[2, 4], y_label='delta(trace_dFF)')

            x_scatter = [float(np.random.rand(1) * 1)] * len(mean_photostim_responses)
            pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses],
                                       ax_titles=[f"{expobj.t_series_name} - photostims"],
                                       figsize=[2, 4], y_label='delta(trace_dFF)')

        else:
            x_scatter = [float(np.random.rand(1) * 1)] * len(mean_photostim_responses)
            pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses],
                                       ax_titles=[expobj.t_series_name],
                                       figsize=[2, 4], y_label='delta(trace_dFF)')
            # pplot.plot_bar_with_points(data=[mean_photostim_responses], bar = False, title=expobj.t_series_name)

    # 3.1) Plotting mean photostim response amplitude across experiments
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=True, set_cache=False)
    def allexps_plot_photostim_responses_magnitude(**kwargs):
        expobj: alloptical = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.plot_photostim_responses_magnitude(expobj=expobj, stims='all')


    # 4) Zscoring of photostimulation responses
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

    # 5) Measuring photostim responses in relation to pre-stim mean FOV Flu
    @staticmethod
    def collect__prestim_FOV_Flu():
        """
        two act function that collects pre-stim FOV Flu value for each stim frame for all experiments, and returns average prestim FOV values across stim group types.

        1) collect pre-stim FOV Flu value for each stim frame. Add these as a var to the expobj.PhotostimResponsesSLMTargets.adata table
            - length of the pre-stim == expobj.pre_stim

        2) collect average prestim FOV values across various stim group types.

        """

        # PART 1)   ####################################################################################################
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
        def __collect_prestim_FOV_Flu_allstims(**kwargs):
            """collect pre-stim Flu from mean raw flu trace and add as a new variable to anndata object."""
            expobj: alloptical = kwargs['expobj']
            pre_stim_FOV_flu = []
            for stim in expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame:
                sli_ce = np.s_[stim - expobj.pre_stim: stim]
                _pre_stim_FOV_flu = expobj.meanRawFluTrace[sli_ce]
                pre_stim_FOV_flu.append(np.round(np.mean(_pre_stim_FOV_flu), 3))

            expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='pre_stim_FOV_Flu', values=pre_stim_FOV_flu)
            expobj.save()

        __collect_prestim_FOV_Flu_allstims()

        # PART 2)   ####################################################################################################
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=False)
        def __collect_prestim_FOV_Flu_pre4ap(**kwargs):
            """Return pre-stim FOV flu for all pre-4ap experiments."""
            expobj: alloptical = kwargs['expobj']

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
            """Return pre-stim FOV flu for interictal and ictal stims from all post-4ap experiments."""

            expobj: Post4ap = kwargs['expobj']

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
                interictal_pre_stim_FOV_flu = expobj.PhotostimResponsesSLMTargets.adata.var.pre_stim_FOV_Flu[
                    interictal_stims_idx]

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


    # 6) measuring photostim responses of targets as a function of pre-stim surrounding signal (targets_annulus Flu)

    """
    1. measuring photostim responses of targets as a function of pre-stim surrounding neuropil signal -- not immediately setup yet to do analysis involving suite2p
    - using approach that sidesteps suite2p is just directly grabbing a annulus of area around the SLM target

    # -- Collect pre-stim frames from all targets_annulus for each stim
    #   -- should result in 3D array of # targets x # stims x # pre-stim frames
    # -- avg above over axis = 2 then add results to anndata object
    """

    # 6.1) collect targets_annulus_prestim_Flu
    def make__targets_annulus_prestim_Flu(self, expobj: Union[alloptical, Post4ap]):
        """
        Average targets_annulus_raw_prestim over axis = 2 then add results to anndata object as a new layer.

        """

        # run procedure to collect and retrieve raw targets_annulus prestim snippets
        expobj.procedure__collect_annulus_data()

        targets_annulus_prestim_rawF = np.mean(expobj.targets_annulus_raw_prestim, axis=2)

        self.adata.add_layer(layer_name='targets_annulus_prestim_rawF', data=targets_annulus_prestim_rawF)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=0)
    def run__targets_annulus_prestim_Flu(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.make__targets_annulus_prestim_Flu(expobj=expobj)
        expobj.save()

    # 6.2) plot targets_annulus_prestim_Flu across stim groups

    """
    - maybe could plot across time on the x axis.

    """

    @staticmethod
    def retrieve__targets_annlus_prestim_Flu():
        """
        Gets pre-stim (from anndata object) the targets_annulus Flu value for each stim frame for all experiments.


        """

        # import alloptical_utils_pj as aoutils
        # expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
        #
        # RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

        # get targets_annulus prestim Flu from anndata #################################################################
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, allow_rerun=0)
        def __targets_annulus_prestim_Flu_pre4ap(**kwargs):
            """Return pre-stim targets annulus for all pre-4ap experiments."""
            expobj: alloptical = kwargs['expobj']

            if 'pre' in expobj.exptype:
                baseline = expobj.PhotostimResponsesSLMTargets.adata.layers['targets_annulus_prestim_rawF']

                return np.round(np.mean(baseline), 3)

        baseline_pre_stim_targets_annulus = __targets_annulus_prestim_Flu_pre4ap()

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
        def __targets_annulus_prestim_Flu_post4ap(**kwargs):
            """Return pre-stim targets annulus for all post-4ap experiments."""
            expobj: alloptical = kwargs['expobj']
            cls_inst = expobj.PhotostimResponsesSLMTargets
            if 'post' in expobj.exptype:
                interictal_stims = cls_inst.adata.layers['targets_annulus_prestim_rawF'][:,
                                   cls_inst.interictal_stims_idx]
                ictal_stims = cls_inst.adata.layers['targets_annulus_prestim_rawF'][:, cls_inst.ictal_stims_idx]

                print('aay')

                return [np.round(np.mean(interictal_stims), 3), np.round(np.mean(ictal_stims), 3)]

        func_collector = __targets_annulus_prestim_Flu_post4ap()

        assert len(func_collector) > 0, '__targets_annulus_prestim_Flu_post4ap didnot return any results.'

        interictal_pre_stim_targets_annulus, ictal_pre_stim_targets_annulus = np.asarray(func_collector)[:,
                                                                              0], np.asarray(
            func_collector)[:, 1]

        # process returned data to make flat arrays
        pre_stim_targets_annulus_results = {'baseline': baseline_pre_stim_targets_annulus,
                                            'interictal': interictal_pre_stim_targets_annulus,
                                            'ictal': ictal_pre_stim_targets_annulus}

        return pre_stim_targets_annulus_results

    @staticmethod
    def plot__targets_annulus_prestim_Flu(RESULTS):
        """
        1. plot Flu of targets_annulus averaged across all targets and stims across baseline, interictal and ictal group types
        - but remember that this should be heterogeneous (especially in the ictal group as targets go from out to in seizure).
        - so might need to consider better ways of plotting this.
        """

        """plot avg pre-stim Flu values across baseline, interictal, and ictal stims"""

        baseline__prestimannulus_flu = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_F['baseline']:
            baseline__prestimannulus_flu.append(np.round(np.mean(exp__prestim_flu), 5))

        interictal__prestimannulus_flu = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_F['interictal']:
            interictal__prestimannulus_flu.append(np.round(np.mean(exp__prestim_flu), 5))

        ictal__prestimannulus_flu = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_F['ictal']:
            ictal__prestimannulus_flu.append(np.round(np.mean(exp__prestim_flu), 5))

        pplot.plot_bar_with_points(
            data=[baseline__prestimannulus_flu, interictal__prestimannulus_flu, ictal__prestimannulus_flu],
            bar=False, x_tick_labels=['baseline', 'interictal', 'ictal'],
            colors=['blue', 'green', 'purple'],
            expand_size_x=0.4, title='Average Pre-stim targets annulus F', y_label='raw F')

    # 6.3) plot targets_annulus_prestim_Flu across targets inside vs. outside sz boundary
    @staticmethod
    def retrieve__targets_annlus_prestim_Flu_duringsz():
        """
        Gets pre-stim (from anndata object) the targets_annulus Flu value for each stim frame for all experiments.


        """

        # import alloptical_utils_pj as aoutils
        # expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
        #
        # RESULTS: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

        # get targets_annulus prestim Flu from anndata #################################################################
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
        def __targets_annulus_prestim_Flu_ictal(**kwargs):
            """Return pre-stim targets annulus for all post-4ap experiments."""
            expobj: alloptical = kwargs['expobj']
            cls_inst = expobj.PhotostimResponsesSLMTargets
            if 'post' in expobj.exptype:
                out_sz_idxs = cls_inst.adata.layers['distance_to_sz'] > 0
                in_sz_idxs = cls_inst.adata.layers['distance_to_sz'] < 0
                out_sz_targets_annulus = out_sz_idxs * cls_inst.adata.layers['targets_annulus_prestim_rawF']
                in_sz_targets_annulus = in_sz_idxs * cls_inst.adata.layers['targets_annulus_prestim_rawF']

                out_sz_targets_annulus_nonzero = np.where(out_sz_targets_annulus != 0, out_sz_targets_annulus, np.nan)
                in_sz_targets_annulus_nonzero = np.where(in_sz_targets_annulus != 0, in_sz_targets_annulus, np.nan)

                return np.round(np.nanmean(out_sz_targets_annulus_nonzero), 3), np.round(
                    np.nanmean(in_sz_targets_annulus_nonzero), 3)

        func_collector = __targets_annulus_prestim_Flu_ictal()

        assert len(func_collector) > 0, '__targets_annulus_prestim_Flu_post4ap didnot return any results.'

        outsz_pre_stim_targets_annulus, insz_pre_stim_targets_annulus = np.asarray(func_collector)[:, 0], np.asarray(
            func_collector)[:, 1]

        # process returned data to make flat arrays
        pre_stim_targets_annulus_results_ictal = {'ictal_outsz': outsz_pre_stim_targets_annulus,
                                                  'ictal_insz': insz_pre_stim_targets_annulus
                                                  }

        return pre_stim_targets_annulus_results_ictal

    @staticmethod
    def plot__targets_annulus_prestim_Flu_outszvsinsz(RESULTS):
        """plot average targets_annulus for targets in ICTAL comparing targets IN SZ and OUT SZ during seizure ICTAL"""

        outsz_pre_stim_targets_annulus = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_results_ictal['ictal_outsz']:
            outsz_pre_stim_targets_annulus.append(np.round(np.mean(exp__prestim_flu), 5))

        insz_pre_stim_targets_annulus = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_results_ictal['ictal_insz']:
            insz_pre_stim_targets_annulus.append(np.round(np.mean(exp__prestim_flu), 5))

        pplot.plot_bar_with_points(
            data=[outsz_pre_stim_targets_annulus, insz_pre_stim_targets_annulus],
            bar=False, x_tick_labels=['out sz', 'in sz'], colors=['orange', 'red'],
            expand_size_x=0.4, title='Average Pre-stim targets annulus F out sz vs. in sz', y_label='raw F')
        pass

    @staticmethod
    def plot__targets_annulus_prestim_Flu_combined(RESULTS):
        """plot average targets_annulus for targets combining baseline, interictal, and separating out IN SZ and OUT SZ"""

        baseline__prestimannulus_flu = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_F['baseline']:
            baseline__prestimannulus_flu.append(np.round(np.mean(exp__prestim_flu), 5))

        interictal__prestimannulus_flu = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_F['interictal']:
            interictal__prestimannulus_flu.append(np.round(np.mean(exp__prestim_flu), 5))

        outsz_pre_stim_targets_annulus = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_results_ictal['ictal_outsz']:
            outsz_pre_stim_targets_annulus.append(np.round(np.mean(exp__prestim_flu), 5))

        insz_pre_stim_targets_annulus = []
        for exp__prestim_flu in RESULTS.expavg_pre_stim_targets_annulus_results_ictal['ictal_insz']:
            insz_pre_stim_targets_annulus.append(np.round(np.mean(exp__prestim_flu), 5))

        pplot.plot_bar_with_points(
            data=[baseline__prestimannulus_flu, interictal__prestimannulus_flu, outsz_pre_stim_targets_annulus,
                  insz_pre_stim_targets_annulus],
            bar=False, x_tick_labels=['baseline', 'interictal', 'outsz', 'insz'],
            colors=['blue', 'green', 'orange', 'red'],
            expand_size_x=0.4, title='Average Pre-stim targets annulus F', y_label='raw F')


    @staticmethod
    def plot__targets_annulus_prestim_Flu2(RESULTS):
        """maybe plot average targets_annulus as a function of the FOV Flu (of individual stims)"""

        pass

    # 6.4) plot targets photostim responses vs. targets_annulus_prestim_Flu
    #  -- plus create version splitting targets inside vs. outside sz boundary; consider including baseline, interictal in same plot
    @staticmethod
    def retrieve__photostim_responses_vs_prestim_targets_annulus_flu():
        """plot avg target photostim responses in relation to pre-stim targets_annulus Flu value across baseline, interictal, and ictal stims.
        - collect average target responses over all stims in a group, and similarly average target_annulus prestim Flu over all stims in a group

        x-axis = pre-stim targets_annulus FOV flu, y-axis = photostim responses"""

        rer = True
        fig, axs = plt.subplots(figsize=(20, 5), nrows=1, ncols=4)

        alpha = 0.1
        size = 30

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=0, set_cache=1, allow_rerun=rer)
        def _collect_data_pre4ap(**kwargs):
            expobj: alloptical = kwargs['expobj']
            ax = kwargs['ax']
            assert 'pre' in expobj.exptype, f'wrong expobj exptype. {expobj.exptype}. expected pre'

            # collect over all targets via average over all stims in the group
            x_data_avg = np.round(
                np.mean(expobj.PhotostimResponsesSLMTargets.adata.layers['targets_annulus_prestim_rawF'], axis=1), 3)
            y_data_avg = np.round(np.mean(expobj.PhotostimResponsesSLMTargets.adata.X, axis=1), 3)

            # collect all datapoints from data array
            x_data_full = pj.flattenOnce(
                np.round(expobj.PhotostimResponsesSLMTargets.adata.layers['targets_annulus_prestim_rawF'], 3))
            y_data_full = pj.flattenOnce(np.round(expobj.PhotostimResponsesSLMTargets.adata.X, 3))

            ax.scatter(x_data_full, y_data_full, facecolor='blue', alpha=alpha, s=size, label='baseline')
            # pj.plot_hist_density(data=[x_data_full], ax=ax, show=False, fill_color=['blue'], fig=fig)
            # ax.hist(x_data_full)
            # sns.kdeplot(x=x_data_full, y=y_data_full, color='cornflowerblue', ax=ax, alpha=0.1)

            # return ax
            return x_data_full, y_data_full

        # axs[0] = _collect_data_pre4ap(ax=axs[0])[-1]

        func_collector = _collect_data_pre4ap(ax=axs[0])

        assert len(func_collector) > 0
        # x_data_full, y_data_full = np.asarray(func_collector)[:, 0], np.asarray(func_collector)[:, 1]  #, np.asarray(func_collector)[:, 2]
        x_data_full, y_data_full = pj.flattenOnce(np.asarray(func_collector)[:, 0]), pj.flattenOnce(
            np.asarray(func_collector)[:, 1])  # <-- this version works

        # sns.kdeplot(x=x_data_full, y=y_data_full, color='cornflowerblue', ax=axs[1], alpha=1)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=1, allow_rerun=rer)
        def _collect_data_post4ap(**kwargs):
            expobj: alloptical = kwargs['expobj']
            ax = kwargs['ax']
            ax2 = kwargs['ax2']
            ax3 = kwargs['ax3']
            assert 'post' in expobj.exptype, f'wrong expobj exptype. {expobj.exptype}. expected post'

            cls_inst: PhotostimResponsesQuantificationSLMtargets = expobj.PhotostimResponsesSLMTargets

            # inter-ictal stims - collect all datapoints from interictal stims #############################################
            interic_targets_annulus = pj.flattenOnce(
                cls_inst.adata.layers['targets_annulus_prestim_rawF'][:, cls_inst.interictal_stims_idx])
            interic_targets_responses = pj.flattenOnce(cls_inst.adata.X[:, cls_inst.interictal_stims_idx])
            ax.scatter(interic_targets_annulus, interic_targets_responses, facecolor='green', alpha=alpha, s=size,
                       label='interictal')
            # sns.kdeplot(interic_targets_annulus, interic_targets_responses, color='forestgreen', ax=ax)

            # # ictal stims - collect all datapoints from ictal stims #############################################
            # x_data_full = pj.flattenOnce(cls_inst.adata.layers['targets_annulus_prestim_rawF'][:, cls_inst.ictal_stims_idx])
            # y_data_full = pj.flattenOnce(cls_inst.adata.X[:, cls_inst.ictal_stims_idx])
            # ax.scatter(x_data_full, y_data_full, facecolor='purple', alpha=0.03, s=50, label='ictal')

            # ictal stims - break up for stims/targets outsz #############################################
            out_sz_idxs = cls_inst.adata.layers['distance_to_sz'] > 0
            out_sz_targets_annulus = cls_inst.adata.layers['targets_annulus_prestim_rawF'][out_sz_idxs]
            out_sz_targets_responses = cls_inst.adata.X[out_sz_idxs]
            # x_data_full = pj.flattenOnce(out_sz_targets_annulus)
            # y_data_full = pj.flattenOnce(out_sz_targets_responses)
            ax2.scatter(out_sz_targets_annulus, out_sz_targets_responses, facecolor='orange', alpha=alpha, s=size,
                        label='out sz')
            # sns.kdeplot(x=out_sz_targets_annulus, y=out_sz_targets_responses, color='orange', ax=ax2)

            # ictal stims - break up for stims/targets insz #############################################
            in_sz_idxs = cls_inst.adata.layers['distance_to_sz'] < 0
            in_sz_targets_annulus = cls_inst.adata.layers['targets_annulus_prestim_rawF'][in_sz_idxs]
            in_sz_targets_responses = cls_inst.adata.X[in_sz_idxs]
            # x_data_full = pj.flattenOnce(in_sz_targets_annulus)
            # y_data_full = pj.flattenOnce(in_sz_targets_responses)
            ax3.scatter(in_sz_targets_annulus, in_sz_targets_responses, facecolor='red', alpha=alpha, s=size,
                        label='in sz')
            # sns.kdeplot(x=in_sz_targets_annulus, y=in_sz_targets_responses, color='tomato', ax=ax2)

            # return ax, ax2, ax3
            return interic_targets_annulus, interic_targets_responses, out_sz_targets_annulus, out_sz_targets_responses, in_sz_targets_annulus, in_sz_targets_responses

        # axs[1:] = _collect_data_post4ap(ax=axs[1], ax2=axs[2], ax3=axs[3])[-1]

        func_collector = _collect_data_post4ap(ax=axs[1], ax2=axs[2], ax3=axs[3])

        assert len(func_collector) > 0
        interic_targets_annulus, interic_targets_responses, out_sz_targets_annulus, out_sz_targets_responses, in_sz_targets_annulus, \
        in_sz_targets_responses = pj.flattenOnce(np.asarray(func_collector)[:, 0]), pj.flattenOnce(
            np.asarray(func_collector)[:, 1]), \
                                  pj.flattenOnce(np.asarray(func_collector)[:, 2]), pj.flattenOnce(
            np.asarray(func_collector)[:, 3]), \
                                  pj.flattenOnce(np.asarray(func_collector)[:, 4]), pj.flattenOnce(
            np.asarray(func_collector)[:, 5])

        pre_stim_targets_annulus_vs_targets_responses_results = {'baseline_targets_annulus': x_data_full,
                                                                 'baseline_targets_responses': y_data_full,
                                                                 'interic_targets_annulus': interic_targets_annulus,
                                                                 'interic_targets_responses': interic_targets_responses,
                                                                 'out_sz_targets_annulus': out_sz_targets_annulus,
                                                                 'out_sz_targets_responses': out_sz_targets_responses,
                                                                 'in_sz_targets_annulus': in_sz_targets_annulus,
                                                                 'in_sz_targets_responses': in_sz_targets_responses}

        # pj.plot_hist_density(data=[x_data_full, interic_targets_annulus, out_sz_targets_annulus, in_sz_targets_annulus], fill_color=['blue', 'green', 'orange', 'red'],
        #                      show_legend=False, legend_labels=['', '', '', ''], figsize=[5,5])
        # pj.plot_hist_density(data=[x_data_full], ax=axs[0], show=False, fig=fig, title='baseline_targets_annulus')
        # pj.plot_hist_density(data=[interic_targets_annulus], ax=axs[1], show=False, fig=fig, title='interic_targets_annulus')
        # pj.plot_hist_density(data=[out_sz_targets_annulus], ax=axs[2], show=False, fig=fig, title='out_sz_targets_annulus')
        # pj.plot_hist_density(data=[in_sz_targets_annulus], ax=axs[3], show=False, fig=fig, title='in_sz_targets_annulus')
        # fig.show()

        # use log scale for x axis
        # axs = pj.flattenOnce(axs)
        titles = ['baseline', 'interictal', 'ictal - out sz', 'ictal - in sz']
        [axs[i].set_title(titles[i], wrap=True) for i in range(len(axs))]
        [axs[i].set_xscale('log') for i in range(len(axs))]
        [axs[i].set_ylim([-100, 100]) for i in range(len(axs))]
        [axs[i].set_xlim([10 ** 1, 10 ** 3]) for i in range(len(axs))]

        # ax.legend(handles_, labels_, loc='center left', bbox_to_anchor=(1.04, 0.5))
        [axs[i].set_xlabel('pre-stim annulus Flu') for i in range(len(axs))]
        [axs[i].set_ylabel('avg dFF of targets') for i in range(len(axs))]
        fig.tight_layout(pad=2)

        # Utils.save_figure(fig, save_path_suffix="plot__pre-stim-fov_vs_avg-photostim-response-of-targets.png")
        fig.suptitle('retrieve__photostim_responses_vs_prestim_targets_annulus_flu')
        fig.show()

        return pre_stim_targets_annulus_vs_targets_responses_results

    @staticmethod
    def plot__photostim_responses_vs_prestim_targets_annulus_flu(RESULTS):
        # import alloptical_utils_pj as aoutils
        # expobj: Post4ap = Utils.import_expobj(prep='PS11', trial='t-011')

        RESULTS = RESULTS.pre_stim_targets_annulus_vs_targets_responses_results

        from _utils_.alloptical_plotting import dataplot_frame_options
        dataplot_frame_options()

        # fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3)
        fig, axs = plt.subplots(figsize=(20, 5), nrows=1, ncols=4)
        # axs = [axs]
        print(f'\nCreating plot `photostim_responses_vs_prestim_targets_annulus_flu`\n')

        alpha = 0.1
        size = 30

        # sns.kdeplot(x=RESULTS['baseline_targets_annulus'], y=RESULTS['baseline_targets_responses'],
        #             color='cornflowerblue', ax=axs[0, 0], alpha=0.4, fill=True)
        # sns.kdeplot(x=RESULTS['interic_targets_annulus'], y=RESULTS['interic_targets_responses'], color='forestgreen',
        #             fill=True, ax=axs[0, 1], alpha=0.4)
        # sns.kdeplot(x=RESULTS['out_sz_targets_annulus'], y=RESULTS['out_sz_targets_responses'], color='orange',
        #             ax=axs[0, 2], alpha=0.4, fill=True)
        # sns.kdeplot(x=RESULTS['in_sz_targets_annulus'], y=RESULTS['in_sz_targets_responses'], color='red', ax=axs[0, 3],
        #             alpha=0.4, fill=True)

        axs[0].scatter(RESULTS['baseline_targets_annulus'], RESULTS['baseline_targets_responses'],
                          facecolor='cornflowerblue', alpha=alpha, s=size, label='baseline')
        axs[1].scatter(RESULTS['interic_targets_annulus'], RESULTS['interic_targets_responses'],
                          facecolor='forestgreen', alpha=alpha, s=size, label='interic')
        axs[2].scatter(RESULTS['out_sz_targets_annulus'], RESULTS['out_sz_targets_responses'], facecolor='orange',
                          alpha=alpha, s=size, label='out_sz')
        axs[3].scatter(RESULTS['in_sz_targets_annulus'], RESULTS['in_sz_targets_responses'], facecolor='red',
                          alpha=alpha, s=size, label='in_sz')

        axs[0].errorbar(np.mean(RESULTS['baseline_targets_annulus']), 
                        np.mean(RESULTS['baseline_targets_responses']), 
                        xerr=np.std(RESULTS['baseline_targets_annulus'], ddof=1), 
                        yerr=np.std(RESULTS['baseline_targets_responses']), 
                        fmt='o', color='black',
                    ecolor='gray', elinewidth=5, capsize=2)
        axs[1].errorbar(np.mean(RESULTS['interic_targets_annulus']), 
                        np.mean(RESULTS['interic_targets_responses']),
                        xerr=np.std(RESULTS['interic_targets_annulus'], ddof=1), 
                        yerr=np.std(RESULTS['interic_targets_responses']), 
                        fmt='o', color='black',
                    ecolor='gray', elinewidth=5, capsize=2)
        axs[2].errorbar(np.mean(RESULTS['out_sz_targets_annulus']), 
                        np.mean(RESULTS['out_sz_targets_responses']), 
                        xerr=np.std(RESULTS['out_sz_targets_annulus'], ddof=1), 
                        yerr=np.std(RESULTS['out_sz_targets_responses']), 
                        fmt='o', color='black',
                    ecolor='gray', elinewidth=5, capsize=2)
        axs[3].errorbar(np.mean(RESULTS['in_sz_targets_annulus']), 
                        np.mean(RESULTS['in_sz_targets_responses']), 
                        xerr=np.std(RESULTS['in_sz_targets_annulus'], ddof=1), 
                        yerr=np.std(RESULTS['in_sz_targets_responses']), 
                        fmt='o', color='black',
                    ecolor='gray', elinewidth=5, capsize=2)

        # plt.show()

        # complete plot
        titles = ['baseline', 'interictal', 'ictal - out sz', 'ictal - in sz']
        # [axs[0, i].set_title(titles[i], wrap=True) for i in range(axs.shape[1])]
        [axs[i].set_title(titles[i], wrap=True) for i in range(axs.shape[0])]

        # # create custom legend
        # handles, labels = ax.get_legend_handles_labels()
        # # labels_ = np.unique(labels)
        # labels_ = ['baseline', 'out sz']
        # handles_ = [handles[labels.index(label)] for label in labels_]

        # import matplotlib.patches as mpatches
        # radius = 3
        # baseline_leg = mpatches.Circle((0,0), radius, color='green', label='baseline')
        # interictal_leg = mpatches.Circle((0,0), radius, color='blue', label='interictal')
        # ictal_leg = mpatches.Circle((0,0), radius, color='purple', label='ictal')
        # in_sz_leg = mpatches.Circle((0,0), radius, color='orange', label='in sz')
        # out_sz_leg = mpatches.Circle((0,0), radius, color='red', label='out sz')
        # handles_custom = [baseline_leg, interictal_leg, ictal_leg, in_sz_leg, out_sz_leg]
        # labels_custom = ['baseline', 'interictal', 'ictal', 'in sz', 'out sz']

        # use log scale for x axis
        # axs = pj.flattenOnce(axs)
        [axs[i].set_xscale('log') for i in range(len(axs))]
        [axs[i].set_ylim([-200, 200]) for i in range(len(axs))]
        [axs[i].set_xlim([10 ** 1, 10 ** 3.5]) for i in range(len(axs))]

        # ax.legend(handles_, labels_, loc='center left', bbox_to_anchor=(1.04, 0.5))
        [axs[i].set_xlabel('pre-stim annulus Flu') for i in range(len(axs))]
        [axs[i].set_ylabel('dFF of targets') for i in range(len(axs))]
        fig.tight_layout(pad=2)
        fig.suptitle('plot__photostim_responses_vs_prestim_targets_annulus_flu')
        # Utils.save_figure(fig, save_path_suffix="plot__pre-stim-fov_vs_avg-photostim-response-of-targets.png")
        fig.show()

    @staticmethod
    def plot__targets_annulus_prestim_Flu_all_points(RESULTS):
        """plot targets_annulus for all targets/stims across exps, plot shows: baseline, interictal, and separating out IN SZ and OUT SZ"""

        RESULTS = RESULTS.pre_stim_targets_annulus_vs_targets_responses_results

        from _utils_.alloptical_plotting import dataplot_frame_options
        dataplot_frame_options()

        pplot.plot_bar_with_points(
            data=[RESULTS['baseline_targets_annulus'], RESULTS['interic_targets_annulus'],
                  RESULTS['out_sz_targets_annulus'],
                  RESULTS['in_sz_targets_annulus']], bar=False,
            x_tick_labels=['baseline', 'interictal', 'outsz', 'insz'],
            colors=['blue', 'green', 'orange', 'red'], expand_size_x=0.4, suptitle='Average Pre-stim targets annulus F',
            y_label='raw F',
            title='plot__targets_annulus_prestim_Flu_all_points', alpha=0.1)


# %% CODE RUNNING ZONE

"""
TODO:

MAJOR
[x] collecting and plotting prestim Flu of annulus around target - goal is to see that rise consistently across targets that in seizure, vs. 
not rise for targets that ARE NOT in seizure. would be really helpful to show and make these plots.
[x] - then also compare photostim responses in relation to the targets' annulus Flu.

MINOR
send out plots for prestim FOV flu and prestim targets annulus flu 

"""


if __name__ == '__main__':
    expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-009')
    # self = expobj.PhotostimResponsesSLMTargets
    # self.add_targets_annulus_prestim_anndata(expobj=expobj)

    # RESULTS.pre_stim_targets_annulus_vs_targets_responses_results = retrieve__photostim_responses_vs_prestim_targets_annulus_flu()
    # RESULTS.save_results()
    # retrieve__photostim_responses_vs_prestim_targets_annulus_flu()
    PhotostimResponsesQuantificationSLMtargets.plot__photostim_responses_vs_prestim_targets_annulus_flu(RESULTS)
    PhotostimResponsesQuantificationSLMtargets.plot__targets_annulus_prestim_Flu_all_points(RESULTS)

    pass

# %% ARCHIVE

# collect pre-stim flu from targets_annulus:
# -- for each SLM target: create a numpy array slice that acts as an annulus around the target
# -- determine slice object for collecting pre-stim frames
# -- read in raw registered tiffs, then use slice object to collect individual targets' annulus raw traces directly from the tiffs
#   -- should result in 3D array of # targets x # stims x # pre-stim frames
# -- avg above over axis = 2 then add results to anndata object

# #####  moved all below to methods under alloptical main. . 22/03/09
# # Collect pre-stim frames from all targets_annulus for each stim
# def _TargetsExclusionZone(self: alloptical, distance: float = 2.5):
#     """
#     creates an annulus around each target of the specified diameter that is considered the exclusion zone around the SLM target.
#
#     # use the SLM targets exclusion zone areas as the annulus around each SLM target
#     # -- for each SLM target: create a numpy array slice that acts as an annulus around the target
#
#
#     :param self:
#     :param distance: distance from the edge of the spiral to extend the target exclusion zone
#
#     """
#
#     distance = 5
#
#     frame = np.zeros(shape=(self.frame_x, self.frame_y), dtype=int)
#
#     # target_areas that need to be excluded when filtering for nontarget cells
#     radius_px_exc = int(np.ceil(((self.spiral_size / 2) + distance) / self.pix_sz_x))
#     print(f"radius of target exclusion zone (in pixels): {radius_px_exc}px")
#
#     target_areas = []
#     for coord in self.target_coords_all:
#         target_area = ([item for item in pj.points_in_circle_np(radius_px_exc, x0=coord[0], y0=coord[1])])
#         target_areas.append(target_area)
#     self.target_areas_exclude = target_areas
#
#     # create annulus by subtracting SLM spiral target pixels
#     radius_px_target = int(np.ceil(((self.spiral_size / 2)) / self.pix_sz_x))
#     print(f"radius of targets (in pixels): {radius_px_target}px")
#
#     target_areas_annulus_all = []
#     for idx, coord in enumerate(self.target_coords_all):
#         target_area = ([item for item in pj.points_in_circle_np(radius_px_target, x0=coord[0], y0=coord[1])])
#         target_areas_annulus = [coord_ for i, coord_ in enumerate(self.target_areas_exclude[idx]) if coord_ not in target_area]
#         target_areas_annulus_all.append(target_areas_annulus)
#     self.target_areas_exclude_annulus = target_areas_annulus_all
#
#     # add to frame_array towards creating a plot
#     for area in self.target_areas_exclude:
#         for x, y in area:
#             frame[x, y] = -10
#
#     for area in self.target_areas_exclude_annulus:
#         for x, y in area:
#             frame[x, y] = 10
#
#     return self.target_areas_exclude_annulus
#     # plt.figure(figsize=(4, 4))
#     # plt.imshow(frame, cmap='BrBG')
#     # plt.show()
#
# # -- determine slice object for collecting pre-stim frames
# def _create_slice_obj_excl_zone(self: alloptical):
#     """
#     creates a list of slice objects for each target.
#
#     :param self:
#     """
#     # frame = np.zeros(shape=(expobj.frame_x, expobj.frame_y), dtype=int)  # test frame
#
#     arr = np.asarray(self.target_areas_exclude_annulus)
#     annulus_slice_obj = []
#     # _test_sum = 0
#     slice_obj_full = np.array([np.array([])] * 2, dtype=int)  # not really finding any use for this, but have it here in case need it
#     for idx, coord in enumerate(self.target_coords_all):
#         annulus_slice_obj_target = np.s_[arr[idx][:, 0], arr[idx][:, 1]]
#         # _test_sum += np.sum(frame[annulus_slice_obj_target])
#         annulus_slice_obj.append(annulus_slice_obj_target)
#         slice_obj_full = np.hstack((slice_obj_full, annulus_slice_obj_target))
#
#     # slice_obj_full = np.asarray(annulus_slice_obj)
#     # frame[slice_obj_full[0, :], slice_obj_full[1, :]]
#
#     return annulus_slice_obj
#
# def _collect_annulus_flu(self: alloptical, annulus_slice_obj):
#     """
#     Read in raw registered tiffs, then use slice object to collect individual targets' annulus raw traces directly from the tiffs
#
#     :param self:
#     :param annulus_slice_obj: list of len(n_targets) containing the numpy slice object for SLM targets
#     """
#
#     print('\n\ncollecting raw Flu traces from SLM target coord. areas from registered TIFFs')
#
#     # read in registered tiff
#     reg_tif_folder = self.s2p_path + '/reg_tif/'
#     reg_tif_list = os.listdir(reg_tif_folder)
#     reg_tif_list.sort()
#     start = self.curr_trial_frames[0] // 2000  # 2000 because that is the batch size for suite2p run
#     end = self.curr_trial_frames[1] // 2000 + 1
#
#     mean_img_stack = np.zeros([end - start, self.frame_x, self.frame_y])
#     # collect mean traces from target areas of each target coordinate by reading in individual registered tiffs that contain frames for current trial
#     targets_annulus_traces = np.zeros([len(self.slmtargets_ids), (end - start) * 2000], dtype='float32')
#     for i in range(start, end):
#         tif_path_save2 = self.s2p_path + '/reg_tif/' + reg_tif_list[i]
#         with TiffFile(tif_path_save2, multifile=False) as input_tif:
#             print('\t reading tiff: %s' % tif_path_save2)
#             data = input_tif.asarray()
#
#         target_annulus_trace = np.zeros([len(self.target_coords_all), data.shape[0]], dtype='float32')
#         for idx, coord in enumerate(self.target_coords_all):
#             # target_areas = np.array(self.target_areas)
#             # x = data[:, target_areas[coord, :, 1], target_areas[coord, :, 0]]
#             x = data[:, annulus_slice_obj[idx][0], annulus_slice_obj[idx][1]]
#             target_annulus_trace[idx] = np.mean(x, axis=1)
#
#         targets_annulus_traces[:, (i - start) * 2000: ((i - start) * 2000) + data.shape[0]] = target_annulus_trace  # iteratively write to each successive segment of the targets_trace array based on the length of the reg_tiff that is read in.
#
#     # final part, crop to the exact frames for current trial
#     self.raw_SLMTargets_annulus = targets_annulus_traces[:, self.curr_trial_frames[0] - start * 2000: self.curr_trial_frames[1] - (start * 2000)]
#
#     return self.raw_SLMTargets_annulus
#
# def retrieve_annulus_prestim_snippets(self: alloptical):
#     """
#     # -- Collect pre-stim frames from all targets_annulus for each stim
#     #   -- should result in 3D array of # targets x # stims x # pre-stim frames
#     """
#
#     stim_timings = self.stim_start_frames
#
#     data_to_process = self.raw_SLMTargets_annulus
#
#     num_targets = len(self.slmtargets_ids)
#     targets_trace = data_to_process
#
#     # collect photostim timed average dff traces of photostim targets
#     targets_annulus_raw_prestim = np.zeros([num_targets, len(self.stim_start_frames), self.pre_stim])
#
#     for targets_idx in range(num_targets):
#         flu = [targets_trace[targets_idx][stim - self.pre_stim: stim] for stim in stim_timings]
#         for i in range(len(flu)):
#             trace = flu[i]
#             targets_annulus_raw_prestim[targets_idx, i] = trace
#
#     self.targets_annulus_raw_prestim = targets_annulus_raw_prestim
#     print(f"Retrieved targets_annulus pre-stim traces for {num_targets} targets, {len(stim_timings)} stims, and {int(self.pre_stim/self.fps)} secs")
#     return targets_annulus_raw_prestim
#
# def procedure__collect_annulus_data(self: alloptical):
#     """
#     Full procedure to define annulus around each target and retrieve data from annulus.
#
#     Read in raw registered tiffs, then use slice object to collect individual targets' annulus raw traces directly from the tiffs
#
#     """
#     self.target_areas_exclude_annulus = _TargetsExclusionZone(self=self)
#     annulus_slice_obj = _create_slice_obj_excl_zone(self=self)
#     _collect_annulus_flu(self=self, annulus_slice_obj=annulus_slice_obj)
#     retrieve_annulus_prestim_snippets(self=self)
