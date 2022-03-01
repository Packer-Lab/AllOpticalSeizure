import os
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

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
save_path = SAVE_LOC + 'TargetsSzInvasionTemporal.pkl'

if os.path.exists(save_path):
    save_TargetsSzInvasionTemporal = pj.load_pkl(pkl_path=save_path)
else:
    save_TargetsSzInvasionTemporal = {}



class TargetsSzInvasionTemporal(Quantification):
    range_of_sz_invasion_time: List[
        float] = None  # TODO need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion time stats calculated across all targets and all exps


    @staticmethod
    def save_data(dataname: str, data):
        save_TargetsSzInvasionTemporal[dataname] = data
        pj.save_pkl(obj=save_TargetsSzInvasionTemporal, pkl_path=save_path)

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\- ADDING NEW TargetsSzInvasionTemporal MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionTemporal <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    ## 1.1) collecting mean of seizure invasion Flu traces from all targets for each experiment ########################
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
            cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if
                     'time_del' in col]
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

    ## 1.2) plot mean of seizure invasion Flu traces from all targets for each experiment ##############################
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

    ## 2) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME  ####################################

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
            cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if
                     'time_del' in col]
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

    ## 2.1) PLOT NUM STIMS PRE AND POST SZ INVASION TOTAL ACROSS ALL TARGETS FOR EACH EXPERIMENT #######################
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

    # 3) COLLECT PHOTOSTIM RESPONSES MAGNITUDE VS. PHOTOSTIM RESPONSES FOR AN EXP  #####################################
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
            sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(
                stim_timesz_df[stim][stim_timesz_df[stim].notnull()])
            sztime_v_photostimresponses[stim]['photostim_responses'] += list(
                expobj.PhotostimResponsesSLMTargets.adata.X[:, idx][stim_timesz_df[stim].notnull()])

            # for targetidx, row in stim_timesz_df.iterrows():
            #     # expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()]
            #     sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(row[row.notnull()])
            #     sztime_v_photostimresponses[stim]['photostim_responses'] += list(expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()])
            #
            # print(len(np.unique(sztime_v_photostimresponses[stim]['time_to_szinvasion'])))
            # print(len(np.unique(sztime_v_photostimresponses[stim]['photostim_responses'])))
            assert len(sztime_v_photostimresponses[stim]['time_to_szinvasion']) == len(
                sztime_v_photostimresponses[stim]['photostim_responses'])
            assert sum(np.isnan(sztime_v_photostimresponses[stim]['time_to_szinvasion'])) == 0, print(
                'something went wrong - found an nan in time to szinvasion')

        self.sztime_v_photostimresponses = sztime_v_photostimresponses

    def return_szinvasiontime_vs_photostimresponses(self):
        x = []
        y = []
        for stim, items in self.sztime_v_photostimresponses.items():
            x += items['time_to_szinvasion']
            y += items['photostim_responses']
        return x, y

    ## 3.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE VS. PHOTOSTIM RESPONSES FOR AN EXP  #####################################
    @plot_piping_decorator(figsize=(4, 4), nrows=1, ncols=1, verbose=False)
    def plot_szinvasiontime_vs_photostimresponses(self, fig=None, ax=None, **kwargs):
        fig, ax = (kwargs['fig'], kwargs['ax']) if fig is None and ax is None else (fig, ax)
        ax.grid(zorder=0)
        x, y = self.return_szinvasiontime_vs_photostimresponses()

        ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.3, ec='white')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)

    ## 4) COLLECT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS ###

    ## 4.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS ####

    ## 5) COLLECT PHOTOSTIM RESPONSES MAGNITUDE (ZSCORED) VS. SEIZURE INVASION TIME FOR EACH TARGET FOR AN EXP ########################################################
    def collect_szinvasiontime_vs_photostimresponses_zscored(self, expobj: Post4ap, zscore_type: str = 'dFF (zscored)'):
        """collects dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
        sztime_v_photostimresponses_zscored = {}
        if not hasattr(self, 'time_del_szinv_stims'):
            stim_timesz_df = expobj.time_del_szinv_stims
        else:
            stim_timesz_df = self.time_del_szinv_stims

        stims_list = list(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame)
        for idx, stim in enumerate(stims_list):
            sztime_v_photostimresponses_zscored[stim] = {'time_to_szinvasion': [],
                                                         'photostim_responses_zscored': []}
            sztime_v_photostimresponses_zscored[stim]['time_to_szinvasion'] += list(
                stim_timesz_df[stim][stim_timesz_df[stim].notnull()])
            sztime_v_photostimresponses_zscored[stim]['photostim_responses_zscored'] += list(
                expobj.PhotostimResponsesSLMTargets.adata.layers[zscore_type][:, idx][stim_timesz_df[stim].notnull()])

            # for targetidx, row in stim_timesz_df.iterrows():
            #     # expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()]
            #     sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(row[row.notnull()])
            #     sztime_v_photostimresponses[stim]['photostim_responses'] += list(expobj.PhotostimResponsesSLMTargets.adata.X[int(targetidx)][row.notnull()])
            #
            # print(len(np.unique(sztime_v_photostimresponses[stim]['time_to_szinvasion'])))
            # print(len(np.unique(sztime_v_photostimresponses[stim]['photostim_responses'])))
            assert len(sztime_v_photostimresponses_zscored[stim]['time_to_szinvasion']) == len(
                sztime_v_photostimresponses_zscored[stim]['photostim_responses_zscored'])
            assert sum(np.isnan(sztime_v_photostimresponses_zscored[stim]['time_to_szinvasion'])) == 0, print(
                'something went wrong - found an nan in time to szinvasion')

        self.sztime_v_photostimresponses_zscored = sztime_v_photostimresponses_zscored

    def collect_szinvasiontime_vs_photostimresponses_zscored_new(self, expobj: Post4ap, zscore_type: str = 'dFF (zscored)'):
        # (re-)make pandas dataframe
        df = pd.DataFrame(columns=['target_id', 'stim_id', 'time_to_szinvasion', 'photostim_responses'])

        stim_ids = [stim for stim in self.time_del_szinv_stims if sum(self.time_del_szinv_stims[stim].notnull()) > 0]

        index_ = 0
        for idx, target in enumerate(expobj.PhotostimResponsesSLMTargets.adata.obs.index):
            for idxstim, stim in enumerate(stim_ids):
                sztime = self.time_del_szinv_stims.loc[target, stim]
                response = expobj.PhotostimResponsesSLMTargets.adata.layers[zscore_type][idx, idxstim]
                if not np.isnan(sztime):
                    df = pd.concat([df, pd.DataFrame({'target_id': target, 'stim_id': stim, 'time_to_szinvasion': sztime,
                                                      'photostim_responses': response}, index=[index_])])  # TODO need to update the idx to use a proper index range
                    index_ += 1

        self.sztime_v_photostimresponses_zscored_df = df


    def return_szinvasiontime_vs_photostimresponses_zscored(self):
        x = []
        y = []
        for stim, items in self.sztime_v_photostimresponses_zscored.items():
            x += items['time_to_szinvasion']
            y += items['photostim_responses_zscored']
        return x, y

    ## 5.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE (ZSCORED) VS. SEIZURE INVASION TIME FOR EACH TARGET FOR AN EXP ##########
    @plot_piping_decorator(figsize=(4, 4), nrows=1, ncols=1, verbose=False)
    def plot_szinvasiontime_vs_photostimresponseszscored(self, fig=None, ax=None, **kwargs):
        fig, ax = (kwargs['fig'], kwargs['ax']) if fig is None and ax is None else (fig, ax)
        ax.grid(zorder=0)
        x, y = self.return_szinvasiontime_vs_photostimresponses()

        ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.3, ec='white')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)


    ## 6.1) PLOT - binning and plotting 2d density plot, and smoothing data across the time to seizure axis, when comparing to responses
    def plot_responses_vs_time_to_seizure_SLMTargets_2ddensity(self, expobj: Post4ap, plot=True):

        print(f"\t|- plotting responses vs. time to seizure")

        sztime_v_photostimresponses_df = self.sztime_v_photostimresponses_zscored_df

        data_expobj = np.array([[], []]).T
        for target in expobj.PhotostimResponsesSLMTargets.adata.obs.index:
            indexes = sztime_v_photostimresponses_df[sztime_v_photostimresponses_df['target_id'] == target].index
            responses = np.array(sztime_v_photostimresponses_df.loc[indexes, 'photostim_responses'])
            time_to_sz = np.asarray(
                sztime_v_photostimresponses_df.loc[indexes, 'time_to_szinvasion'])

            _data = np.array([time_to_sz, responses]).T

            data_expobj = np.vstack((_data, data_expobj))

        times_to_sz = data_expobj[:, 0]
        bin_size = 20  # um
        # bins_num = int((max(times_to_sz) - min(times_to_sz)) / bin_size)
        bins_num = 40

        pj.plot_hist2d(data=data_expobj, bins=[bins_num,bins_num], y_label='dFF (zscored)', title=expobj.t_series_name,
                       figsize=(4, 2), x_label='time to seizure (sec)',
                       y_lim=[-2, 2]) if plot else None

        return data_expobj

    ## TODO BINNING AND SMOOTHING TIME TO SZ INVASION DATA FOR PLOTTING #############################################

    ## 6.2) COLLECT (times represented in percentile space) binning and plotting density plot, and smoothing data across the time to seizure axis, when comparing to responses
    @staticmethod
    def convert_responses_sztimes_percentile_space(data):
        """converts sz invasion times to percentile space"""
        data_all = np.array([[], []]).T
        for data_ in data:
            data_all = np.vstack((data_, data_all))

        from scipy.stats import percentileofscore

        times_to_sz = data_all[:, 0]
        idx_sorted = np.argsort(times_to_sz)
        times_to_sz_sorted = times_to_sz[idx_sorted]
        responses_sorted = data_all[:, 1][idx_sorted]
        s = pd.Series(times_to_sz_sorted)
        percentiles = s.apply(lambda x: percentileofscore(times_to_sz_sorted, x))
        scale_percentile_times = {}
        for pct in range(0, 100):
            scale_percentile_times[int(pct + 1)] = np.round(np.percentile(times_to_sz_sorted, pct), 0)
        data_all = np.array([percentiles, responses_sorted]).T

        return data_all, percentiles, responses_sorted, times_to_sz_sorted, scale_percentile_times

    @staticmethod
    def plot_density_responses_sztimes(data_all, times_to_sz_sorted, scale_percentile_times):
        # plotting density plot for all exps, in percentile space (to normalize for excess of data at times which are closer to zero) - TODO any smoothing?

        bin_size = 2  # secs
        # bins_num = int((max(times_to_sz) - min(times_to_sz)) / bin_size)
        bins_num = [100, 500]

        fig, ax = plt.subplots(figsize=(6, 3))
        pj.plot_hist2d(data=data_all, bins=bins_num, y_label='dFF (zscored)', figsize=(6, 3),
                       x_label='time to seizure (%tile space)',
                       title=f"2d density plot, all exps, 50%tile = {np.percentile(times_to_sz_sorted, 50)}um",
                       y_lim=[-3, 3], fig=fig, ax=ax, show=False)
        ax.axhline(0, ls='--', c='white', lw=1)
        xticks = [1, 25, 50, 57, 75, 100]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_times[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel('time to seizure (um)')

        fig.show()

    # plotting line plot for all datapoints for responses vs. time to seizure
    @staticmethod
    def plot_lineplot_responses_pctsztimes(percentiles, responses_sorted, response_type,
                                           scale_percentile_times):
        percentiles_binned = np.round(percentiles)

        bin = 5
        # change to pct% binning
        percentiles_binned = (percentiles_binned // bin) * bin

        d = {'time to seizure (%tile space)': percentiles_binned,
             response_type: responses_sorted}

        df = pd.DataFrame(d)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=df, x='time to seizure (%tile space)', y=response_type, ax=ax)
        ax.set_title(f'responses over time to sz, all exps, normalized to percentile space ({bin}% bins)',
                     wrap=True)
        ax.margins(0.02)
        ax.axhline(0, ls='--', c='orange', lw=1)

        xticks = [1, 25, 50, 57, 75, 100]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_times[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel('time to seizure (um)')

        fig.tight_layout(pad=2)
        plt.show()


if __name__ == '__main__':
    main = TargetsSzInvasionTemporal
    main.data = main.plot_responses_vs_time_to_seizure_SLMTargets_2ddensity()
    pass


# %% ARCHIVE

### below should be almost all refactored to TargetsSzInvasionTemporal class


# %% 3) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME

# create a df containing delay to sz invasion for each target for each stim frame (dim: n_targets x n_stims)
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def collect_time_delay_sz_stims(self, **kwargs):
    expobj: Post4ap = kwargs['expobj']
    df = pd.DataFrame(columns=expobj.PhotostimResponsesSLMTargets.adata.var['stim_start_frame'],
                      index=expobj.slmtargets_data.obs.index)
    for target in expobj.slmtargets_data.obs.index:
        # target = 0
        print(f'\- collecting from target #{target} ... ') if (int(target) % 10) == 0 else None
        cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
        sz_times = expobj.slmtargets_data.obs.iloc[int(target), cols_]
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
    self._update_expobj(expobj)
    expobj.save()


# collect_time_delay_sz_stims()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
def check_collect_time_delay_sz_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    # plot num stims outside of sz invasion and num stims after sz invasion
    pos_values_collect = []
    neg_values_collect = []
    for idx, row in expobj.time_del_szinv_stims.iterrows():
        values = row[row.notnull()]
        pos_values_collect += list(values[values > 0])
        neg_values_collect += list(values[values < 0])

    expobj.time_delay_sz_stims_pos_values_collect = pos_values_collect
    expobj.time_delay_sz_stims_neg_values_collect = neg_values_collect

    print(f"avg num stim before sz invasion: {len(pos_values_collect) / expobj.n_targets_total}")
    print(f"avg num stim after sz invasion: {len(neg_values_collect) / expobj.n_targets_total}")


# check_collect_time_delay_sz_stims()

@plot_piping_decorator(figsize=(8, 4), nrows=1, ncols=1, verbose=False)
def plot_time_delay_sz_stims(**kwargs):
    fig, ax, expobj = kwargs['fig'], kwargs['ax'], kwargs['expobj']

    ax.hist(expobj.time_delay_sz_stims_pos_values_collect, fc='blue')
    ax.hist(expobj.time_delay_sz_stims_neg_values_collect, fc='purple')


# %% 1) collecting mean of seizure invasion Flu traces from all targets for each experiment

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def check_targets_sz_invasion_time(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    print(expobj.slmtargets_data.obs)


# check_targets_sz_invasion_time()


# create dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)
@Utils.run_for_loop_across_exps(run_pre4ap_trials=False,
                                run_post4ap_trials=True)  # , run_trials=['RL109 t-016', 'PS07 t-011', 'PS11 t-011'])
def collect_targets_sz_invasion_traces(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    fov_traces_ = []
    traces_ = []
    pre = 5
    post = 10
    for target, coord in enumerate(expobj.slmtargets_data.obs['SLM target coord']):
        # target, coord = 0, expobj.slmtargets_data.obs['SLM target coord'][0]
        print(f'\- collecting from target #{target} ... ') if (target % 10) == 0 else None
        cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
        sz_times = expobj.slmtargets_data.obs.iloc[target, cols_]
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
    expobj.mean_targets_szinvasion_trace = {'fov_mean_trace': FOV_MEAN_TRACE,
                                            'mean_trace': MEAN_TRACE,
                                            'pre_sec': pre,
                                            'post_sec': post}
    expobj.save()


# collect_targets_sz_invasion_traces()

# %% 2) PLOT TARGETS FLU PERI- TIME TO INVASION

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


# plot_targets_sz_invasion_meantraces_full()


if __name__ == '__main__':
    # collect_time_delay_sz_stims()
    # check_collect_time_delay_sz_stims()
    # plot_time_delay_sz_stims()
    #
    # check_targets_sz_invasion_time()
    # collect_targets_sz_invasion_traces()
    pass
