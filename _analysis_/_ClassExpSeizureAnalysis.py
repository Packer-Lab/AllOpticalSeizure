# NOTE: ALOT OF THIS CODE IS PRIMARILY COPIED AND REFACTORED OVER FROM alloptical_sz_processing IN AN EFFORT TO class-ify THE ANALYSIS OF THE SZ PROCESSING. COPIED OVER BITS ARE ARCHIVED UNDER THAT ORIGINAL SCRIPT.

from typing import List

import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
import funcsforprajay.funcs as pj
import tifffile as tf

from _analysis_._utils import Quantification
from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze, ExpMetainfo, OnePhotonStimExpsToAnalyze
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import multi_plot_subplots, _get_ax_for_multi_plot, plot_SLMtargets_Locs

# %%
from _utils_.io import import_expobj


class ExpSeizureAnalysis(Quantification):
    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\t\- ADDING NEW ExpSeizureAnalysis MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"ExpSeizureAnalysis <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # %% 1.0) calculate time delay between LFP onset of seizures and imaging FOV invasion for each seizure for each experiment

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def FOVszInvasionTime(**kwargs):
        """
        The general approach to calculate seizure invasion time delay (for imaging FOV) is to calculate the first stim
        (which are usually every 10 secs) which has the seizure wavefront in the FOV relative to the LFP onset of the seizure
        (which is at the 4ap inj. site).

        :param kwargs: no args taken. used only to pipe in experiments from for loop.

        """

        expobj: Post4ap = kwargs['expobj']
        time_delay_sec = [-1] * len(expobj.stim_start_frames)
        sz_num = [-1] * len(expobj.stim_start_frames)
        for i in range(expobj.numSeizures):
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

        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='delay_from_sz_onset_sec',
                                                               values=time_delay_sec)
        expobj.PhotostimResponsesSLMTargets.adata.add_variable(var_name='seizure_num', values=sz_num)
        expobj.save()

    # 1.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion

    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
    def plot_sz_invasion(**kwargs):
        expobj: Post4ap = kwargs['expobj']

        sz_nums = np.unique([i for i in list(expobj.slmtargets_data.var.seizure_num) if type(i) is int and i > 0])
        fig, axs, counter, ncols, nrows = multi_plot_subplots(num_total_plots=len(sz_nums))
        for sz in sz_nums:
            idx = np.where(expobj.slmtargets_data.var.seizure_num == sz)[0][0]  # first seizure invasion frame
            stim_frm = expobj.slmtargets_data.var.stim_start_frame[idx]
            time_del = expobj.slmtargets_data.var.delay_from_sz_onset_sec[idx]

            # plotting
            avg_stim_img_path = f'{expobj.analysis_save_path[:-1]}avg_stim_images/{expobj.metainfo["date"]}_{expobj.metainfo["trial"]}_stim-{stim_frm}.tif'
            bg_img = tf.imread(avg_stim_img_path)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_insz, cells=in_sz, edgecolors='yellowgreen', background=bg_img)
            # aoplot.plot_SLMtargets_Locs(self, targets_coords=coords_to_plot_outsz, cells=out_sz, edgecolors='white', background=bg_img)
            ax = _get_ax_for_multi_plot(axs, counter, ncols)
            fig, ax = plot_SLMtargets_Locs(expobj, fig=fig, ax=ax,
                                           title=f"sz #: {sz}, stim_fr: {stim_frm}, time inv.: {time_del}s",
                                           show=False, background=bg_img)

            try:
                inframe_coord1_x = expobj.slmtargets_data.var["seizure location"][idx][0][0]
                inframe_coord1_y = expobj.slmtargets_data.var["seizure location"][idx][0][1]
                inframe_coord2_x = expobj.slmtargets_data.var["seizure location"][idx][1][0]
                inframe_coord2_y = expobj.slmtargets_data.var["seizure location"][idx][1][1]
                ax.plot([inframe_coord1_x, inframe_coord2_x], [inframe_coord1_y, inframe_coord2_y], c='darkorange',
                        linestyle='dashed', alpha=1, lw=2)
            except TypeError:
                print('hitting nonetype error')

        fig.suptitle(f"{expobj.t_series_name} {expobj.date}")
        fig.show()

    plot_sz_invasion()

    # %% 4.1) counting seizure incidence across all imaging trials
    @staticmethod
    def count_sz_incidence_2p_trials():
        for key in list([*AllOpticalExpsToAnalyze.trial_maps['post']]):
            # import initial expobj
            expobj = import_expobj(aoresults_map_id=f'pre {key}.0', verbose=False)
            prep = expobj.metainfo['animal prep.']
            # look at all run_post4ap_trials trials in expobj and for loop over all of those run_post4ap_trials trials
            for trial in expobj.metainfo['post4ap_trials']:
                # import expobj
                expobj = import_expobj(prep=prep, trial=trial, verbose=False)
                total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

                # count seizure incidence (avg. over mins) for each experiment (animal)
                if hasattr(expobj, 'seizure_lfp_onsets'):
                    n_seizures = len(expobj.seizure_lfp_onsets)
                else:
                    n_seizures = 0

                print(f'Seizure incidence for {prep}, {trial}, {expobj.metainfo["exptype"]}: ',
                      np.round(n_seizures / total_time_recording, 2))

    # 4.1.1) measure seizure incidence across onePstim trials
    @staticmethod
    def count_sz_incidence_1p_trials():
        for exp_prep in ExpMetainfo.onephotonstim.post_4ap_trials:
            expobj = import_expobj(exp_prep=exp_prep, verbose=False)
            total_time_recording = np.round((expobj.n_frames / expobj.fps) / 60., 2)  # return time in mins

            # count seizure incidence (avg. over mins) for each experiment (animal)
            if hasattr(expobj, 'seizure_lfp_onsets'):
                n_seizures = len(expobj.seizure_lfp_onsets)
            else:
                n_seizures = 0

            print('Seizure incidence for %s, %s, %s: ' % (
                expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                  np.round(n_seizures / total_time_recording, 2))

    # 4.1.2) plot seizure incidence across onePstim and twoPstim trials
    twop_trials_sz_incidence = [0.35, 0.251666667, 0.91, 0.33, 0.553333333, 0.0875, 0.47, 0.33, 0.52]  # sz/min
    onep_trials_sz_incidence = [0.38, 0.26, 0.19, 0.436666667, 0.685]  # sz/min

    @classmethod
    def plot__sz_incidence(cls):
        pj.plot_bar_with_points(data=[cls.twop_trials_sz_incidence, cls.onep_trials_sz_incidence],
                                x_tick_labels=['2p stim', '1p stim'],
                                colors=['purple', 'green'], y_label='sz incidence (events/min)',
                                title='rate of seizures during exp', expand_size_x=0.4, expand_size_y=1, ylims=[0, 1],
                                shrink_text=0.8)

        pj.plot_bar_with_points(data=[cls.twop_trials_sz_incidence + cls.onep_trials_sz_incidence],
                                x_tick_labels=['Experiments'],
                                colors=['#2E3074'], y_label='Seizure incidence (events/min)', alpha=0.7, bar=False,
                                title='rate of seizures during exp', expand_size_x=0.7, expand_size_y=1, ylims=[0, 1],
                                shrink_text=0.8)

    # %% 4.2) measure seizure LENGTHS across all imaging trials (including any spont imaging you might have)

    @staticmethod
    def count_sz_lengths_2p_trials():
        for key in list([*AllOpticalExpsToAnalyze.trial_maps['post']]):
            # import initial expobj
            expobj = import_expobj(aoresults_map_id=f'pre {key}.0', verbose=False)
            prep = expobj.metainfo['animal prep.']
            # look at all run_post4ap_trials trials in expobj
            # if 'post-4ap trials' in expobj.metainfo.keys():
            #     a = 'post-4ap trials'
            # elif 'post4ap_trials' in expobj.metainfo.keys():
            #     a = 'post4ap_trials'
            # for loop over all of those run_post4ap_trials trials
            for trial in expobj.metainfo['post4ap_trials']:
                # import expobj
                expobj = import_expobj(prep=prep, trial=trial, verbose=False)
                # count the average length of each seizure
                if hasattr(expobj, 'seizure_lfp_onsets'):
                    n_seizures = len(expobj.seizure_lfp_onsets)
                    counter = 0
                    sz_lengths_total = 0
                    if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                        for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                            if sz_onset != 0:
                                sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                                     expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                                counter += 1
                        avg_len = sz_lengths_total / counter
                        expobj.avg_sz_len = avg_len

                        print('Avg. seizure length (secs) for %s, %s, %s: ' % (prep, trial, expobj.metainfo['exptype']),
                              np.round(expobj.avg_sz_len, 2))

                else:
                    n_seizures = 0
                    print('no sz events for %s, %s, %s ' % (prep, trial, expobj.metainfo['exptype']))

    # 4.2.1) measure seizure LENGTHS across onePstim trials
    @staticmethod
    def count_sz_lengths_1p_trials():
        for exp_prep in ExpMetainfo.onephotonstim.post_4ap_trials:
            expobj = import_expobj(exp_prep=exp_prep, verbose=False)
            # count the average length of each seizure
            if hasattr(expobj, 'seizure_lfp_onsets'):
                n_seizures = len(expobj.seizure_lfp_onsets)
                counter = 0
                sz_lengths_total = 0
                if len(expobj.seizure_lfp_onsets) == len(expobj.seizure_lfp_offsets) > 1:
                    for i, sz_onset in enumerate(expobj.seizure_lfp_onsets):
                        if sz_onset != 0:
                            sz_lengths_total += (expobj.frame_clock_actual[expobj.seizure_lfp_offsets[i]] -
                                                 expobj.frame_clock_actual[sz_onset]) / expobj.paq_rate
                            counter += 1
                    avg_len = sz_lengths_total / counter
                    expobj.avg_sz_len = avg_len
                    print('Avg. seizure length (secs) for %s, %s, %s: ' % (
                        expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']),
                          np.round(expobj.avg_sz_len, 2))

            else:
                n_seizures = 0
                print('Avg. seizure length (secs) for %s, %s, %s ' % (
                    expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype']))

    # 4.2.2) plot seizure length across onePstim and twoPstim trials
    twop_trials_sz_lengths = [24.0, 93.73, 38.86, 84.77, 17.16, 83.78, 15.78, 36.88]
    onep_trials_sz_lengths = [30.02, 34.25, 114.53, 35.57]

    @classmethod
    def plot__sz_lengths(cls):
        pj.plot_bar_with_points(data=[cls.twop_trials_sz_lengths, cls.onep_trials_sz_lengths], x_tick_labels=['2p stim', '1p stim'],
                                colors=['purple', 'green'], y_label='seizure length (secs)',
                                title='Avg. length of sz', expand_size_x=0.4, expand_size_y=1, ylims=[0, 120], title_pad=15,
                                shrink_text=0.8)

        pj.plot_bar_with_points(data=[cls.twop_trials_sz_lengths + cls.onep_trials_sz_lengths], x_tick_labels=['Experiments'],
                                colors=['green'], y_label='Seizure length (secs)', alpha=0.7, bar=False,
                                title='Avg sz length', expand_size_x=0.7, expand_size_y=1, ylims=[0, 120],
                                shrink_text=0.8)


