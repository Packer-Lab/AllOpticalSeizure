from typing import List

import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
import funcsforprajay.funcs as pj

from _analysis_._utils import Quantification
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames


class TargetsSzInvasionTemporal(Quantification):
    range_of_sz_invasion_time: List[float] = None  # TODO need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion time stats calculated across all targets and all exps

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\- ADDING NEW TargetsSzInvasionTemporal MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionTemporal <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"




    # %% 1.1) collecting mean of seizure invasion Flu traces from all targets for each experiment
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
    def check_targets_sz_invasion_time(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        print(expobj.PhotostimResponsesSLMTargets.adata.obs)


    def collect_targets_sz_invasion_traces(self, expobj: Post4ap):
        """create dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)
        """
        fov_traces_ = []
        traces_ = []
        pre = 5
        post = 10
        for target, coord in enumerate(expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord']):
            # target, coord = 0, expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord'][0]
            print(f'\- collecting from target #{target} ... ') if (target % 10) == 0 else None
            cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if 'time_del' in col]
            sz_times = expobj.PhotostimResponsesSLMTargets.adata.obs.iloc[target, cols_]
            fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if
                        not pd.isnull(time)]

            # collect each frame seizure invasion time Flu snippet for current target
            target_traces = []
            fov_traces = []
            for fr in fr_times:
                target_tr = expobj.raw_SLMTargets[target][fr - int(pre * expobj.fps): fr + int(post * expobj.fps)]
                fov_tr = expobj.meanRawFluTrace[fr - int(pre * expobj.fps): fr + int(post * expobj.fps)]
                target_traces.append(target_tr)
                fov_traces.append(fov_tr)
                # ax.plot(pj.moving_average(to_plot, n=4), alpha=0.2)
            traces_.append(np.mean(target_traces, axis=0)) if len(target_traces) > 1 else None
            fov_traces_.append(np.mean(fov_traces, axis=0)) if len(target_traces) > 1 else None

        traces_ = np.array(traces_)
        fov_traces_ = np.array(fov_traces_)
        MEAN_TRACE = np.mean(traces_, axis=0)
        FOV_MEAN_TRACE = np.mean(fov_traces_, axis=0)
        self.mean_targets_szinvasion_trace = {'fov_mean_trace': FOV_MEAN_TRACE,
                                                'mean_trace': MEAN_TRACE,
                                                'pre_sec': pre,
                                                'post_sec': post}

    # %% 1.2) plot mean of seizure invasion Flu traces from all targets for each experiment
    @staticmethod
    def plot_targets_sz_invasion_meantraces_full():
        fig, ax = plt.subplots(figsize=[3, 6])

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
        def plot_targets_sz_invasion_meantraces(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            fov_mean_trace, to_plot, pre, post = expobj.mean_targets_szinvasion_trace['fov_mean_trace'], \
                                                 expobj.mean_targets_szinvasion_trace['mean_trace'], \
                                                 expobj.mean_targets_szinvasion_trace['pre_sec'], \
                                                 expobj.mean_targets_szinvasion_trace['post_sec']
            invasion_spot = int(pre * expobj.fps)
            to_plot = pj.moving_average(to_plot, n=6)
            fov_mean_trace = pj.moving_average(fov_mean_trace, n=6)
            to_plot_normalize = (to_plot - to_plot[0]) / to_plot[0]
            fov_mean_normalize = (fov_mean_trace - fov_mean_trace[0]) / fov_mean_trace[0]

            x_time = np.linspace(-pre, post, len(to_plot))

            ax.plot(x_time, to_plot_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3)
            # ax2.plot(x_time, fov_mean_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3, alpha=0.5, linestyle='--')

            ax.scatter(x=0, y=to_plot_normalize[invasion_spot], color='crimson', s=45, zorder=5)

            xticks = [-pre, 0, post]
            # xticks_loc = [xtick*expobj.fps for xtick in [0, pre, pre+post]]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            ax.set_xlabel('Time (secs) to sz invasion')
            ax.set_ylabel('Flu change (norm.)')
            # ax.set_xlim([np.min(xticks_loc), np.max(xticks_loc)])
            # ax.set_title(f"{expobj.t_series_name}")

        plot_targets_sz_invasion_meantraces()
        fig.suptitle('avg Flu at sz invasion', wrap=True, y=0.96)
        fig.tight_layout(pad=2)
        fig.show()

    # %% 2) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME  ##################################################################################################

    def collect_time_delay_sz_stims(self, expobj: Post4ap):
        """

        :param expobj: Post4ap object
        :return: dataframe containing relative time (secs) to sz invasion for each stim and each target
        """
        df = pd.DataFrame(columns=expobj.PhotostimResponsesSLMTargets.adata.var['stim_start_frame'],
                          index=expobj.PhotostimResponsesSLMTargets.adata.obs.index)
        for target in expobj.PhotostimResponsesSLMTargets.adata.obs.index:
            # target = 0
            print(f'\- collecting from target #{target} ... ') if (int(target) % 10) == 0 else None
            cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if 'time_del' in col]
            sz_times = expobj.PhotostimResponsesSLMTargets.adata.obs.iloc[int(target), cols_]
            from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames
            fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if
                        not pd.isnull(time)]
            for szstart, szstop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
                fr_time = [i for i in fr_times if szstart < i < szstop]
                if len(fr_time) >= 1:
                    for stim in expobj.stim_start_frames:
                        if szstart < stim < szstop:
                            time_del = - (stim - fr_time[0]) / expobj.fps
                            df.loc[target, stim] = round(time_del, 2)  # secs

        self.time_del_szinv_stims = df

    def collect_num_pos_neg_szinvasion_stims(self, expobj: Post4ap):
        # collect num stims outside of sz invasion and num stims after sz invasion
        pos_values_collect = []
        neg_values_collect = []
        for idx, row in expobj.time_del_szinv_stims.iterrows():
            values = row[row.notnull()]
            pos_values_collect += list(values[values > 0])
            neg_values_collect += list(values[values < 0])

        self.time_delay_sz_stims_pos_values_collect = pos_values_collect
        self.time_delay_sz_stims_neg_values_collect = neg_values_collect

        print(f"avg num stim before sz invasion: {len(pos_values_collect) / expobj.n_targets_total}")
        print(f"avg num stim after sz invasion: {len(neg_values_collect) / expobj.n_targets_total}")

    # 2.1) PLOT NUM STIMS PRE AND POST SZ INVASION TOTAL ACROSS ALL TARGETS FOR EACH EXPERIMENT
    @plot_piping_decorator(figsize=(8, 4), nrows=1, ncols=1, verbose=False)
    def plot_num_pos_neg_szinvasion_stims(self, **kwargs):
        fig, ax = kwargs['fig'], kwargs['ax']

        ax.hist(self.time_delay_sz_stims_pos_values_collect, fc='blue', ec='red', histtype="stepfilled", alpha=0.5)
        ax.hist(self.time_delay_sz_stims_neg_values_collect, fc='purple', ec='red', histtype="stepfilled", alpha=0.5)

        ax.set_xlabel('Time to sz invasion (secs)')
        ax.set_ylabel('num photostims')
        ax.set_title('num stims, all targets, indiv exps', wrap=1)
        fig.tight_layout(pad=0.8)

        return fig, ax

    # %% 3) COLLECT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS  ########################################################
    def collect_szinvasiontime_vs_photostimresponses(self, expobj: Post4ap):
        """collects dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
        sztime_v_photostimresponses = {}
        if not hasattr(self, 'time_del_szinv_stims'):
            stim_timesz_df = expobj.time_del_szinv_stims
        else:
            stim_timesz_df = self.time_del_szinv_stims

        stims_list = list(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame)
        for idx, stim in enumerate(stims_list):
            sztime_v_photostimresponses[stim] = {'time_to_szinvasion': [],
                                                 'photostim_responses': []}
            sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(stim_timesz_df[stim][stim_timesz_df[stim].notnull()])
            sztime_v_photostimresponses[stim]['photostim_responses'] += list(expobj.PhotostimResponsesSLMTargets.adata.X[:, idx][stim_timesz_df[stim].notnull()])

            # for targetidx, row in stim_timesz_df.iterrows():
            #     # expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()]
            #     sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(row[row.notnull()])
            #     sztime_v_photostimresponses[stim]['photostim_responses'] += list(expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()])
            #
            # print(len(np.unique(sztime_v_photostimresponses[stim]['time_to_szinvasion'])))
            # print(len(np.unique(sztime_v_photostimresponses[stim]['photostim_responses'])))
            assert len(sztime_v_photostimresponses[stim]['time_to_szinvasion']) == len(sztime_v_photostimresponses[stim]['photostim_responses'])
            assert sum(np.isnan(sztime_v_photostimresponses[stim]['time_to_szinvasion'])) == 0, print('something went wrong - found an nan in time to szinvasion')

        self.sztime_v_photostimresponses = sztime_v_photostimresponses

    def return_szinvasiontime_vs_photostimresponses(self):
        x = []
        y = []
        for stim, items in self.sztime_v_photostimresponses.items():
            x += items['time_to_szinvasion']
            y += items['photostim_responses']
        return x, y

    # 3.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS  ########################################################
    @plot_piping_decorator(figsize=(4, 4), nrows=1, ncols=1, verbose=False)
    def plot_szinvasiontime_vs_photostimresponses(self, fig=None, ax=None, **kwargs):
        fig, ax = (kwargs['fig'], kwargs['ax']) if fig is None and ax is None else (fig, ax)
        ax.grid(zorder=0)
        x, y = self.return_szinvasiontime_vs_photostimresponses()

        ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.3, ec='white')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)



    # %% 4) COLLECT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS


    # 4.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS
