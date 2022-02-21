from typing import List

import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
import funcsforprajay.funcs as pj

from _analysis_._utils import Quantification
from _main_.Post4apMain import Post4ap


class ExpSeizureAnalysis(Quantification):
    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\t\- ADDING NEW ExpSeizureAnalysis MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionTemporal <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # %% 1.0) calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def szInvasionTime(**kwargs):
        """
        The general approach to calculate seizure invasion time delay is to calculate the first stim (which are usually every 10 secs)
        which has the seizure wavefront in the FOV relative to the LFP onset of the seizure (which is at the 4ap inj site).

        :param kwargs: no args taken. used only to pipe in experiments from for loop.
        """

        expobj: Post4ap = kwargs['expobj']
        time_delay_sec = [-1] * len(expobj.stim_start_frames)
        sz_num = [-1] * len(expobj.stim_start_frames)
        for i in range(expobj.numSeizures):
            # i = 1
            lfp_onset_fr = expobj.seizure_lfp_onsets[i]
            if lfp_onset_fr != 0:
                start = expobj.seizure_lfp_onsets[i]
                stop = expobj.seizure_lfp_offsets[i]
                _stim_insz = [stim_fr for stim_fr in expobj.stim_start_frames if start < stim_fr < stop]
                stims_wv = [stim_fr for stim_fr in _stim_insz if stim_fr in expobj.stimsWithSzWavefront]
                stims_nowv = [stim_fr for stim_fr in _stim_insz if stim_fr not in expobj.stimsWithSzWavefront]
                if len(stims_wv) > 0:
                    for stim in stims_wv:
                        if stim in expobj.stimsWithSzWavefront:
                            sz_start_sec = start / expobj.fps
                            _time_delay_sec = (stim / expobj.fps) - sz_start_sec
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = round(_time_delay_sec, 3)
                            sz_num[idx] = i
                    for stim in stims_nowv:
                        if stim < stims_wv[0]:  # first in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "bf invasion"  # before seizure invasion to the FOV
                            sz_num[idx] = i
                        elif stim > stims_wv[-1]:  # last in seizure stim frame with the seizure wavefront
                            idx = np.where(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame == stim)[0][0]
                            time_delay_sec[idx] = "af invasion"  # after seizure wavefront has passed the FOV
                            sz_num[idx] = i

        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='delay_from_sz_onset_sec', values=time_delay_sec)
        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='seizure_num', values=sz_num)
        expobj.save()


