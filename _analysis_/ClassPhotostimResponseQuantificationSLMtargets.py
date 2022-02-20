## misc analysis steps
from dataclasses import dataclass
from typing import Union, List

import numpy as np
import pandas as pd

import _alloptical_utils as Utils
from _analysis_._utils import Quantification
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"

# %%
# expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')


# %% COLLECT AND PLOT PHOTOSTIM RESPONSES MAGNITUDES

class PhotostimResponsesQuantificationSLMtargets(Quantification):
    mean_photostim_responses_baseline: List[float] = [-1]
    mean_photostim_responses_interictal: List[float] = [-1]
    mean_photostim_responses_ictal: List[float] = [-1]

    def __init__(self, expobj: alloptical):
        self._metainfo = expobj.metainfo
        print(f'\- ADDING NEW PhotostimResponsesSLMTargets MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"PhotostimResponsesSLMTargets <-- Analysis submodule for expobj <{self.expobj_id}>"

    @classmethod
    def saveclass(cls):
        analysis_save_path = '/home/pshah/mnt/qnap/Analysis/allopticalseizures/PhotostimResponsesQuantificationSLMtargets.pkl'
        from _utils_.io import save_cls_pkl
        save_cls_pkl(cls, analysis_save_path)


    @property
    def expobj_id(self):
        return f"{self._metainfo['animal prep.']} {self._metainfo['trial']}"

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
            expobj.get_SLMTarget_responses_dff(process='delta(trace_dFF)', threshold=10, stims_to_use=expobj.stim_start_frames)


        f, ax = pplot.make_general_scatter(x_list=[np.random.random(self.responses_SLMtargets_tracedFF.shape[0])], y_data=[np.mean(self.responses_SLMtargets_tracedFF, axis=1)],
                                           ax_titles=[f"{expobj.t_series_name}"], show=False, y_label='delta(trace_dff)', figsize=[2,4],
                                           x_lim=[-1,2], y_lim=[-50, 100])
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
                    print(f'******* No slmtargets_szboundary_stim (sz boundary classification not done) for: {expobj.t_series_name}', ' [*2.3] ')




    def create_anndata_SLMtargets(self, expobj: Union[alloptical, Post4ap]):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the Ca2+ matrix of the imaging trial.

        """

        if hasattr(expobj, 'dFF_SLMTargets') or hasattr(expobj, 'raw_SLMTargets'):
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
            photostim_responses_adata = AnnotatedData2(X=photostim_responses, obs=obs_meta, var=var_meta.T, obsm=obs_m, layers=layers, data_label=_data_type)

            print(f"Created: {photostim_responses_adata}")
            self.adata = photostim_responses_adata
        else:
            Warning('did not create anndata. anndata creation only available if experiments were processed with suite2p and .paq file(s) provided for temporal synchronization')


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
        x_scatter = [float(np.random.rand(1)*1) for i in mean_photostim_responses]
        pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses], ax_titles=[expobj.t_series_name],
                                   figsize=[2, 4], y_label='delta(trace_dFF)')
        # pplot.plot_bar_with_points(data=[mean_photostim_responses], bar = False, title=expobj.t_series_name)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1, run_post4ap_trials=1, allow_rerun=True)
    def allexps_plot_photostim_responses_magnitude(**kwargs):
        expobj: alloptical = kwargs['expobj']
        expobj.PhotostimResponsesSLMTargets.plot_photostim_responses_magnitude(expobj=expobj, stims='all')


        # return mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal


