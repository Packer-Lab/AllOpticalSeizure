## misc analysis steps
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
from _analysis_._utils import Quantification
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"

# %%
expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')


# %% COLLECT AND PLOT PHOTOSTIM RESPONSES MAGNITUDES

class PhotostimResponsesQuantificationSLMtargets(Quantification):

    def __init__(self):
        pass


    def collect_photostim_responses_exp(self, expobj: Union[alloptical, Post4ap]):
        """
        runs calculations of photostim responses, calculating reliability of photostim of slm targets,
        saving success stim locations, and saving stim response magnitudes as pandas dataframe.
        - of various methods -

        :param expobj: experiment trial object

        """

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
                        # var_meta.loc['seizure location', fr_idx] = '..not-set-yet..'
                        var_meta.loc['seizure location', fr_idx] = (
                            expobj.stimsSzLocations.coord1[stim_frame], expobj.stimsSzLocations.coord2[stim_frame])
                    else:
                        var_meta.loc['wvfront in sz', fr_idx] = False
                        var_meta.loc['seizure location', fr_idx] = None
                var_meta.loc['stim_start_frame', fr_idx] = stim_frame
                var_meta.loc['im_time_secs', fr_idx] = stim_frame * expobj.fps

            # BUILD LAYERS TO ADD TO anndata OBJECT
            layers = {'SLM Targets photostim responses (dF/stdF)': self.responses_SLMtargets_dfstdf,
                      'SLM Targets photostim responses (dF/prestimF)': self.responses_SLMtargets_dfprestimf
                      }

            # SET PRIMARY DATA
            _data_type = 'SLM Targets photostim responses delta(tracedFF)'
            expobj.responses_SLMtargets_tracedFF.columns = range(len(expobj.stim_start_frames))
            photostim_responses = self.responses_SLMtargets_tracedFF

            print(f"\n\----- CREATING annotated data object using AnnData:")
            # create anndata object
            photostim_responses_adata = AnnotatedData2(X=photostim_responses, obs=obs_meta, var=var_meta.T, obsm=obs_m, layers=layers,
                                   data_label=_data_type)

            print(f"\n{photostim_responses_adata}")
            expobj.photostim_responses_adata = photostim_responses_adata
            expobj.save()
        else:
            Warning(
                'did not create anndata. anndata creation only available if experiments were processed with suite2p and .paq file(s) provided for temporal synchronization')


    @staticmethod
    def collect_photostim_responses_magnitude(expobj: alloptical, stims: Union[slice, str, list] = None):
        assert expobj.slmtargets_data
        if not stims or stims == 'all': stims = slice(0, expobj.slmtargets_data.n_vars)

        mean_photostim_responses = []
        for target in expobj.slmtargets_data.obs.index:
            target = int(target)
            mean_photostim_response = np.mean(expobj.slmtargets_data.X[target, stims])
            mean_photostim_responses.append(mean_photostim_response)
        return mean_photostim_responses

    @classmethod
    def plot_photostim_responses_magnitude(cls, expobj: alloptical, stims: Union[slice, str, list] = None):
        """quick plot of photostim responses of expobj's targets across all stims"""
        mean_photostim_responses = cls.collect_photostim_responses_magnitude(expobj, stims)
        x_scatter = [float(np.random.rand(1)*1) for i in mean_photostim_responses]
        pplot.make_general_scatter(x_list=[x_scatter], y_data=[mean_photostim_responses], ax_titles=[expobj.t_series_name],
                                   figsize=[2,4], y_label='delta(trace_dFF)')
        # pplot.plot_bar_with_points(data=[mean_photostim_responses], bar = False, title=expobj.t_series_name)

    @classmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, ignore_cache=True)
    def allexps_plot_photostim_responses_magnitude(cls, **kwargs):
        expobj = kwargs['expobj']
        cls.plot_photostim_responses_magnitude(expobj=expobj, stims='all')
    # allexps_plot_photostim_responses_magnitude()

    @classmethod
    def full_plot_mean_responses_magnitudes(cls):
        "create plot of mean photostim responses magnitudes for all three exp groups"
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, ignore_cache=True)
        def pre4apexps_collect_photostim_responses(**kwargs):
            expobj: alloptical = kwargs['expobj']
            if 'pre' in expobj.exptype:
                # all stims
                mean_photostim_responses = cls.collect_photostim_responses_magnitude(expobj, stims='all')
                return np.mean(mean_photostim_responses)
        mean_photostim_responses_baseline = pre4apexps_collect_photostim_responses()

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, ignore_cache=True)
        def post4apexps_collect_photostim_responses(**kwargs):
            expobj: Post4ap = kwargs['expobj']
            if 'post' in expobj.exptype:
                # interictal stims
                mean_photostim_responses_interictal = cls.collect_photostim_responses_magnitude(expobj, stims=expobj.stim_idx_outsz)

                # ictal stims
                mean_photostim_responses_ictal = cls.collect_photostim_responses_magnitude(expobj, stims=expobj.stim_idx_insz)

                return np.mean(mean_photostim_responses_interictal), np.mean(mean_photostim_responses_ictal)

        func_collector = post4apexps_collect_photostim_responses()
        mean_photostim_responses_interictal, mean_photostim_responses_ictal = np.asarray(func_collector)[:,0], np.asarray(func_collector)[:,1]

        def plot_photostim_response_of_groups(baseline, interictal, ictal):
            pplot.plot_bar_with_points(data=[baseline, interictal, ictal], x_tick_labels=['baseline', 'interictal', 'ictal'], bar=False,
                                       figsize=[4,5])

        plot_photostim_response_of_groups(mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal)

        # return mean_photostim_responses_baseline, mean_photostim_responses_interictal, mean_photostim_responses_ictal


