import os
import pickle

import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator
from matplotlib import pyplot as plt
import seaborn as sns

import _alloptical_utils as Utils
import funcsforprajay.funcs as pj
import funcsforprajay.plotting as pplot

from _analysis_._utils import Quantification, Results
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"
SAVE_PATH = SAVE_LOC + 'TargetsSzInvasionTemporal.pkl'

# setting temporal sz invasion adjust amount (in secs): note that this adjustment factor is being used globally for all targets + traces for a particular exp.
ADJUST_SZ_INV = {'PS04 t-018': 2,
                 # 'PS11 t-011': 5,
                 'PS11 t-011': 0}  # - changing PS11 t-011 back because most targets are already up against the lfp onset temporally

# %%

class TargetsSzInvasionTemporal(Quantification):

    # archived in favor of a dedicated results class .22/03/07
    # @staticmethod
    # def save_data(dataname: str, data):
    #     save_TargetsSzInvasionTemporal[dataname] = data
    #     pj.save_pkl(obj=save_TargetsSzInvasionTemporal, pkl_path=SAVE_PATH)

    # photostim_responses_zscore_type = 'dFF (zscored) (interictal)'
    photostim_responses_zscore_type = 'dFF (zscored) (baseline)'
    plot_szinvasion_trace_params = {'pre_sec': 5,
                                    'post_sec': 10}  #: # of seconds for collecting trace pre and post sz invasion point for each target for each seizure. used for collecting the trace snippets for subsequent plotting.
    EXCLUDE_TRIAL = [
        'PS11 t-011'  # I think there’s good cause for exclusion from this analysis, since the seizure isnt particularly propagating across temporally (like the cells seem to be recruited right away..) - there’s basically no datapoints prior to sz invasion..
    ]

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\- ADDING NEW TargetsSzInvasionTemporal MODULE to expobj: {expobj.t_series_name}')
        # self.add_slmtargets_time_delay_sz_data(expobj=expobj)  ## changing data struct to private adata for current temporal analysis

    def __repr__(self):
        return f"TargetsSzInvasionTemporal <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0)
    # ADD SLM TARGETS TIME DELAY TO SZ DATA TO expobj
    def add_slmtargets_time_delay_sz_data(self, expobj: Post4ap):
        """
        note restructured this function currently into the _create_anndata for this class's own anndata.

        :param expobj:
        :return:
        """
        csv_path = f'/home/pshah/mnt/qnap/Analysis/analysis_export/slmtargets_time_delay_sz__{expobj.prep}_{expobj.trial}.csv'
        slmtargets_time_delay_sz = pd.read_csv(csv_path)
        expobj.slmtargets_time_delay_sz = slmtargets_time_delay_sz
        print(f"adding slmtargets time delay to {expobj.t_series_name}")
        for column in slmtargets_time_delay_sz.columns[1:]:
            expobj.PhotostimResponsesSLMTargets.adata.add_observation(obs_name=column,
                                                                      values=slmtargets_time_delay_sz[column])
            print('\tPhotostimResponsesSLMTargets.adata.obs: ', expobj.PhotostimResponsesSLMTargets.adata.obs)
        expobj.save()

    # 1) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME  ####################################
    def _create_anndata(self, expobj: Post4ap, time_del_szinv_stims):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object primarily based around time to sz invasion for each target/stim.
        :param expobj:
        :param time_del_szinv_stims:

        """

        obs_meta = pd.DataFrame(
            columns=['SLM group #', 'SLM target coord'], index=range(expobj.n_targets_total))
        for target_idx, coord in enumerate(expobj.target_coords_all):
            for groupnum, coords in enumerate(expobj.target_coords):
                if coord in coords:
                    obs_meta.loc[target_idx, 'SLM group #'] = groupnum
                    obs_meta.loc[target_idx, 'SLM target coord'] = coord
                    break

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
        _data_type = 'SLM Targets - time to sz wavefront'  # primary data label

        # setup time_del_szinv_stims with correct column names
        time_del_szinv_stims.columns = var_meta.columns

        print(f"\t\----- CREATING annotated data object using AnnData:")
        # create anndata object
        from _utils_._anndata import AnnotatedData2
        time_del_tosz_array_adata = AnnotatedData2(X=time_del_szinv_stims, obs=obs_meta, var=var_meta.T, data_label=_data_type)

        print(f"Created: {time_del_tosz_array_adata}")
        self.adata = time_del_tosz_array_adata

        # add time to sz invasion for targets across seizures as adata obs
        csv_path = f'/home/pshah/mnt/qnap/Analysis/analysis_export/slmtargets_time_delay_sz__{expobj.prep}_{expobj.trial}.csv'
        slmtargets_time_delay_sz = pd.read_csv(csv_path)
        expobj.slmtargets_time_delay_sz = slmtargets_time_delay_sz
        print(f"adding slmtargets time delay to {expobj.t_series_name} temporal sz invasion adata")
        for column in slmtargets_time_delay_sz.columns[1:]:
            self.adata.add_observation(obs_name=column, values=slmtargets_time_delay_sz[column])
            print('\t.adata.obs: ', self.adata.obs.keys())
        expobj.save()


    def collect_time_delay_sz_stims(self, expobj: Post4ap):
        """
        Convert time stamps of sz invasion for each SLM target in each seizure, into time delay (secs) to sz invasion from sz onset.
        Note: positive time delay values  == sz invasion is forth coming. negative time delay values == sz invasion has passed.

        Also create an anndata object on the basis of the time delay values at each stims.

        Note: stims outside of sz, or targets where sz does not invade are NaN values.

        :param expobj: Post4ap object

        :return: dataframe containing relative time (secs) to sz invasion for each stim and each target
        """
        df = pd.DataFrame(columns=expobj.PhotostimResponsesSLMTargets.adata.var['stim_start_frame'],
                          index=expobj.PhotostimResponsesSLMTargets.adata.obs.index)

        csv_path = f'/home/pshah/mnt/qnap/Analysis/analysis_export/slmtargets_time_delay_sz__{expobj.prep}_{expobj.trial}.csv'
        slmtargets_time_delay_sz = pd.read_csv(csv_path)

        # cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if 'time_del' in col]
        cols_ = [idx for idx, col in enumerate([*slmtargets_time_delay_sz]) if 'time_del' in col]
        for target in expobj.PhotostimResponsesSLMTargets.adata.obs.index:
            # target = 47
            print(f'\- collecting from target # {target}/{expobj.PhotostimResponsesSLMTargets.adata.n_obs} ... ') if (int(target) % 10) == 0 else None
            # sz_times = expobj.PhotostimResponsesSLMTargets.adata.obs.iloc[int(target), cols_]
            sz_times = slmtargets_time_delay_sz.iloc[int(target), cols_]
            fr_times = [TargetsSzInvasionTemporal._convert_timedel2frames(expobj, sznum, time) for sznum, time in
                        enumerate(sz_times) if not pd.isnull(time)]
            for szstart, szstop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
                fr_time = [i for i in fr_times if szstart < i < szstop]
                if len(fr_time) == 1:
                    for stim in expobj.stim_start_frames:
                        if szstart < stim < szstop:
                            time_del = - (stim - fr_time[0]) / expobj.fps
                            df.loc[target, stim] = round(time_del, 2)  # secs
        self.time_del_szinv_stims = df

        self._create_anndata(expobj=expobj, time_del_szinv_stims=self.time_del_szinv_stims)


    ## 1.1) COLLECT AND PLOT NUM STIMS PRE AND POST SZ INVASION TOTAL ACROSS ALL TARGETS FOR EACH EXPERIMENT #######################
    def collect_num_pos_neg_szinvasion_stims(self, expobj: Post4ap):
        """collect num stims outside of sz invasion and num stims after sz invasion"""

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

    @plot_piping_decorator(figsize=(8, 4), nrows=1, ncols=1, verbose=False)
    def plot_num_pos_neg_szinvasion_stims(self, **kwargs):
        """plot num stims outside of sz invasion and num stims after sz invasion"""

        fig, ax = kwargs['fig'], kwargs['ax']

        ax.hist(self.time_delay_sz_stims_pos_values_collect, fc='blue', ec='red', histtype="stepfilled", alpha=0.5)
        ax.hist(self.time_delay_sz_stims_neg_values_collect, fc='purple', ec='red', histtype="stepfilled", alpha=0.5)

        ax.set_xlabel('Time to sz invasion (secs)')
        ax.set_ylabel('num photostims')
        ax.set_title('num stims, all targets, indiv exps', wrap=1)
        fig.tight_layout(pad=0.8)

        return fig, ax

    # 2) collecting mean of seizure invasion Flu traces from all targets for each experiment ########################

    @staticmethod
    def _convert_timedel2frames(expobj: Post4ap, sz_num: int, timestamp: float):
        """
        Converts a time delay value (for sz invasion for a given target) to an absolute frame number from imaging.

        :param expobj:
        :param sz_num:
        :param timestamp:
        :return:
        """
        start, stop = expobj.seizure_lfp_onsets[sz_num], expobj.seizure_lfp_offsets[sz_num]  # start and end frame numbers for the given sz_num

        # apply adjustment factor based on above defined temporal sz invasion adjust amount (in secs):
        _adjust_factor = 0  # probably want to use this in add_slmtargets_time_delay_sz_data or upstream after determining if this route is viable
        if expobj.t_series_name in [*ADJUST_SZ_INV]:
            _adjust_factor_sec = ADJUST_SZ_INV[expobj.t_series_name]
            _adjust_factor = _adjust_factor_sec * expobj.fps

        numFrames = timestamp * expobj.fps
        frameNumber = numFrames + start - _adjust_factor

        return int(round(frameNumber))


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, skip_trials=EXCLUDE_TRIAL)
    def check_targets_sz_invasion_time(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        print(expobj.PhotostimResponsesSLMTargets.adata.obs)

    def collect_targets_sz_invasion_traces(self, expobj: Post4ap, pre_sec=plot_szinvasion_trace_params['pre_sec'],
                                           post_sec=plot_szinvasion_trace_params['post_sec']):
        """create dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)

        :param expobj: Post4ap experiment object
        :param pre_sec: # of seconds for collecting trace pre sz invasion point for each target for each seizure
        :param post_sec: # of seconds for collecting trace post sz invasion point for each target for each seizure
        """

        print(f"|- collecting indiv. sz invasion traces for each target ... ")
        fov_traces_ = []
        traces_ = []
        pre = pre_sec
        post = post_sec
        for target, coord in enumerate(expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord']):
            # target, coord = 0, expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord'][0]
            print(f'\t\- collecting from target # {target}/{expobj.PhotostimResponsesSLMTargets.adata.n_obs} ... ') if (target % 10) == 0 else None
            # cols_ = [idx for idx, col in enumerate([*expobj.PhotostimResponsesSLMTargets.adata.obs]) if
            #          'time_del' in col]
            cols_ = [idx for idx, col in enumerate([*self.adata.obs]) if
                     'time_del' in col]
            sz_times = self.adata.obs.iloc[target, cols_]
            fr_times = [TargetsSzInvasionTemporal._convert_timedel2frames(expobj, sznum, time) for sznum, time in
                        enumerate(sz_times) if
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
        _mean_trace = np.mean(traces_, axis=0)
        _fov_mean_trace = np.mean(fov_traces_, axis=0)
        self.mean_targets_szinvasion_trace = {'fov_mean_trace': _fov_mean_trace,
                                              'mean_trace': _mean_trace,
                                              'pre_sec': pre,
                                              'post_sec': post}

        print(f"|- collected {traces_.shape} traces.")



    # 3) COLLECT PHOTOSTIM RESPONSES MAGNITUDE VS. PHOTOSTIM RESPONSES FOR AN EXP  #####################################
    def collect_szinvasiontime_vs_photostimresponses(self, expobj: Post4ap):
        """collects dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj"""
        sztime_v_photostimresponses = {}
        assert hasattr(self, 'time_del_szinv_stims')  # TODO this shouldn't be needed anymore, all expobj's should have time_del_szinv_stims in the Temporal submodule
        # stim_timesz_df = self.time_del_szinv_stims
        stim_timesz_df = pd.DataFrame(self.adata.X)  # sz invasion time delay for each target at all stims


        stims_list = list(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame)
        for idx, stim in enumerate(stims_list):
            # general approach: for each stim, collect target indexes where the time delay value is not NaN. use these indexes to retrieve the appropriate Photostim response value from the .PhotostimResponsesSLMTargets.adata frame.

            sztime_v_photostimresponses[stim] = {'time_to_szinvasion': [],
                                                 'photostim_responses': []}

            targets_to_pick = stim_timesz_df[idx].notnull()

            sztime_v_photostimresponses[stim]['time_to_szinvasion'] += list(
                stim_timesz_df[idx][targets_to_pick])
            sztime_v_photostimresponses[stim]['photostim_responses'] += list(
                expobj.PhotostimResponsesSLMTargets.adata.X[:, idx][targets_to_pick])

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

    # 3.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE VS. PHOTOSTIM RESPONSES FOR AN EXP  #####################################
    def return_szinvasiontime_vs_photostimresponses(self):
        """
        retrieve list of datapoints for sz invasion time (return in x), and datapoints for photostim responses

        :return:
        """
        x = []
        y = []
        for stim, items in self.sztime_v_photostimresponses.items():
            x += items['time_to_szinvasion']
            y += items['photostim_responses']
        return x, y  ##### HERE RIGHT NOW

    @plot_piping_decorator(figsize=(6, 3), nrows=1, ncols=1, verbose=False)
    def plot_szinvasiontime_vs_photostimresponses(self, fig=None, ax=None, **kwargs):
        fig, ax = (kwargs['fig'], kwargs['ax']) if fig is None and ax is None else (fig, ax)
        ax.grid(zorder=0)
        x, y = self.return_szinvasiontime_vs_photostimresponses()

        pplot.make_general_scatter(x_list=[x], y_data=[y], facecolors = ['red'], edgecolors = ['white'], ax_y_labels=['dFF response'],
                                ax_x_labels=['time delay to sz invasion (secs)'], ax_titles = [f'{self.expobj_id}'],
                                fig=fig, ax=ax, alpha=0.3)

        # ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.3, ec='white')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)

    # 4) COLLECT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS ###
    ## 4.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE PRE AND POST SEIZURE INVASION FOR EACH TARGET ACROSS ALL EXPERIMENTS ####

    # 5) COLLECT PHOTOSTIM RESPONSES MAGNITUDE (ZSCORED) VS. SEIZURE INVASION TIME FOR EACH TARGET FOR AN EXP ########################################################
    def collect_szinvasiontime_vs_photostimresponses_zscored(self, expobj: Post4ap):
        """
        collects dictionary of sz invasion time and photostim responses across all targets for each stim for an expobj.
        uses the class var specified in self.photostim_responses_zscore_type to select the zscored photostim responses type.

        """
        sztime_v_photostimresponses_zscored = {}
        assert hasattr(self,
                       'time_del_szinv_stims')  # TODO this shouldn't be needed anymore, all expobj's should have time_del_szinv_stims in the Temporal submodule

        # stim_timesz_df = self.time_del_szinv_stims
        stim_timesz_df = pd.DataFrame(self.adata.X)  # sz invasion time delay for each target at all stims

        stims_list = list(expobj.PhotostimResponsesSLMTargets.adata.var.stim_start_frame)
        for idx, stim in enumerate(stims_list):
            sztime_v_photostimresponses_zscored[stim] = {'time_to_szinvasion': [],
                                                         'photostim_responses_zscored': []}

            targets_to_pick = stim_timesz_df[idx].notnull()

            sztime_v_photostimresponses_zscored[stim]['time_to_szinvasion'] += list(
                stim_timesz_df[idx][targets_to_pick])
            sztime_v_photostimresponses_zscored[stim]['photostim_responses_zscored'] += list(
                expobj.PhotostimResponsesSLMTargets.adata.layers[self.photostim_responses_zscore_type][:, idx][targets_to_pick])

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

    def collect_szinvasiontime_vs_photostimresponses_zscored_df(self, expobj: Post4ap):
        """collects sz invasion time for each target/stim, and stores as pandas dataframes."""

        # (re-)make pandas dataframe
        df = pd.DataFrame(columns=['target_id', 'stim_id', 'time_to_szinvasion', 'photostim_responses'])

        stim_ids = [stim for stim in self.time_del_szinv_stims if sum(self.time_del_szinv_stims[stim].notnull()) > 0]

        index_ = 0
        for idx, target in enumerate(expobj.PhotostimResponsesSLMTargets.adata.obs.index):
            for idxstim, stim in enumerate(stim_ids):
                sztime = self.time_del_szinv_stims.loc[target, stim]
                response = expobj.PhotostimResponsesSLMTargets.adata.layers[self.photostim_responses_zscore_type][idx, idxstim]
                # response = expobj.PhotostimResponsesSLMTargets.adata.X[idx, idxstim]  # temporarlly switching to dFF responses
                if not np.isnan(sztime):
                    df = pd.concat(
                        [df, pd.DataFrame({'target_id': target, 'stim_id': stim, 'time_to_szinvasion': sztime,
                                           'photostim_responses': response},
                                          index=[index_])])  # TODO need to update the idx to use a proper index range
                    index_ += 1

        self.sztime_v_photostimresponses_zscored_df = df

    def return_szinvasiontime_vs_photostimresponses_zscored(self):
        x = list(self.sztime_v_photostimresponses_zscored_df['time_to_szinvasion'])
        y = list(self.sztime_v_photostimresponses_zscored_df['photostim_responses'])

        # x = []
        # y = []
        # for stim, items in self.sztime_v_photostimresponses_zscored.items():
        #     x += items['time_to_szinvasion']
        #     y += items['photostim_responses_zscored']
        return x, y

    ## 5.1) PLOT PHOTOSTIM RESPONSES MAGNITUDE (ZSCORED) VS. SEIZURE INVASION TIME FOR EACH TARGET FOR AN EXP ##########
    @plot_piping_decorator(figsize=(6, 3), nrows=1, ncols=1, verbose=False)
    def plot_szinvasiontime_vs_photostimresponseszscored(self, fig=None, ax=None, **kwargs):
        fig, ax = (kwargs['fig'], kwargs['ax']) if fig is None and ax is None else (fig, ax)
        ax.grid(zorder=0)
        x, y = self.return_szinvasiontime_vs_photostimresponses_zscored()

        pplot.make_general_scatter(x_list=[x], y_data=[y], facecolors = ['red'], edgecolors = ['white'], ax_y_labels=[f'{self.photostim_responses_zscore_type}'],
                                ax_x_labels=['time delay to sz invasion (secs)'], ax_titles = [f'{self.expobj_id}'],
                                fig=fig, ax=ax, alpha=0.3)

        # ax.scatter(x, y, s=50, zorder=3, c='red', alpha=0.3, ec='white')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='purple', zorder=5, lw=4)

    # 6.1) PLOT - binning and plotting 2d density plot, and smoothing data across the time to seizure axis, when comparing to responses
    def retrieve__responses_vs_time_to_seizure_SLMTargets_plot2ddensity(self, expobj: Post4ap, plot=True):
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

        bins_num = 40
        pj.plot_hist2d(data=data_expobj, bins=[bins_num, bins_num], y_label=self.photostim_responses_zscore_type, title=expobj.t_series_name,
                       figsize=(4, 2), x_label='time to seizure (sec)',
                       y_lim=[-2, 2]) if plot else None

        return data_expobj

    ## TODO BINNING AND SMOOTHING TIME TO SZ INVASION DATA FOR PLOTTING #############################################

    ## 6.2) COLLECT (times represented in percentile space) binning and plotting density plot, and smoothing data across the time to seizure axis, when comparing to responses
    @staticmethod
    def convert_responses_sztimes_percentile_space(data):
        """converts sz invasion times to percentile space. returns sorted lists of data according to percentile space sorting.

        TODO: add shuffled comparisons for control comparisons.
        """
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
    def plot_density_responses_sztimes(data_all, times_to_sz_sorted, scale_percentile_times,
                                       photostim_responses_zscore_type=photostim_responses_zscore_type):
        # plotting density plot for all exps, in percentile space (to normalize for excess of data at times which are closer to zero) - TODO any smoothing?

        bin_size = 2  # secs
        # bins_num = int((max(times_to_sz) - min(times_to_sz)) / bin_size)
        bins_num = [100, 500]

        fig, ax = plt.subplots(figsize=(6, 3))
        pj.plot_hist2d(data=data_all, bins=bins_num, y_label=photostim_responses_zscore_type, figsize=(6, 3),
                       x_label='time to seizure (secs; normalized in pct. space)',
                       title=f"2d density plot, all exps, 50%tile = {np.percentile(times_to_sz_sorted, 50)}secs",
                       y_lim=[-3, 3], fig=fig, ax=ax, show=False)
        ax.axhline(0, ls='--', c='white', lw=1)
        xticks = [1, 25, 50, 57, 75, 100]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_times[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        # ax.set_xlabel('time to seizure (secs)')

        fig.show()

    # plotting line plot for all datapoints for responses vs. time to seizure
    @staticmethod
    def plot_lineplot_responses_pctsztimes(percentiles, responses_sorted, scale_percentile_times,
                                           response_type=photostim_responses_zscore_type, bin=5):
        percentiles_binned = np.round(percentiles)

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
        ax.set_xlabel('time to seizure (secs; normalized in pct. space)')

        fig.tight_layout(pad=2)
        plt.show()


    # 7) time to sz invasion vs. photostim responses - no percentile normalization of time bins
    @staticmethod
    def collect__binned__szinvtime_v_responses():
        """collect time vs. respnses for time bins"""
        bin_width = 3  # sec
        bins = np.arange(-60, 0, bin_width)  # -60 --> 0 secs, split in bins
        num = [0 for _ in range(len(bins))]  # num of datapoints in binned sztemporalinv
        y = [0 for _ in range(len(bins))]  # avg responses at distance bin
        responses = [[] for _ in range(len(bins))]  # collect all responses at distance bin

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=False,
                                        skip_trials=TargetsSzInvasionTemporal.EXCLUDE_TRIAL)
        def add_time_responses(bins, num, y, responses, **kwargs):
            expobj = kwargs['expobj']

            temporal = expobj.TargetsSzInvasionTemporal

            # temporal.sztime_v_photostimresponses

            for _, row in temporal.sztime_v_photostimresponses_zscored_df.iterrows():
                time = row['time_to_szinvasion']
                response = row['photostim_responses']
                for i, bin in enumerate(bins[:-1]):
                    if bins[i] < time < (bins[i + 1]):
                        num[i] += 1
                        y[i] += response
                        responses[i].append(response)

            return num, y, responses

        func_collector = add_time_responses(bins=bins, num=num, y=y, responses=responses)

        num, y, responses = func_collector[-1][0], func_collector[-1][1], func_collector[-1][2]

        sztemporalinv = bins + bin_width / 2

        avg_responses = [np.mean(responses_) for responses_ in responses]

        # calculate 95% ci for avg responses
        import scipy.stats as stats

        conf_int = np.array(
            [stats.t.interval(alpha=0.95, df=len(responses_) - 1, loc=np.mean(responses_), scale=stats.sem(responses_))
             for responses_ in responses])

        results.binned__time_vs_photostimresponses = {'bin_width_sec': bin_width, 'sztemporal_bins': sztemporalinv,
                                                      'num_points_in_bin': num,
                                                      'avg_photostim_response_in_bin': avg_responses,
                                                      '95conf_int': conf_int}

        results.save_results()

        # return bin_width, sztemporalinv, num,  avg_responses, conf_int

    @staticmethod
    def plot__responses_v_szinvtemporal_no_normalization(results, **kwargs):
        """plotting of binned responses over time to sz invasion for each target+stim as a step function, with heatmap showing # of datapoints"""
        # sztemporalinv_bins = results.binned__distance_vs_photostimresponses['sztemporal_bins']
        sztemporalinv = results.binned__time_vs_photostimresponses['sztemporal_bins']
        avg_responses = results.binned__time_vs_photostimresponses['avg_photostim_response_in_bin']
        conf_int = results.binned__time_vs_photostimresponses['95conf_int']
        num2 = results.binned__time_vs_photostimresponses['num_points_in_bin']

        conf_int_sztemporalinv = pj.flattenOnce(
            [[sztemporalinv[i], sztemporalinv[i + 1]] for i in range(len(sztemporalinv) - 1)])
        conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
        conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])

        fig, axs = (kwargs['fig'], kwargs['axes']) if 'fig' in kwargs or 'axes' in kwargs else plt.subplots(figsize=(6, 5), nrows=2, ncols=1, dpi=300)
        # ax.plot(sztemporalinv[:-1], avg_responses, c='cornflowerblue', zorder=1)
        ax = axs[0][0] if 'fig' in kwargs or 'axes' in kwargs else axs[0]
        ax2 = axs[1][0] if 'fig' in kwargs or 'axes' in kwargs else axs[1]
        ax.step(sztemporalinv, avg_responses, c='cornflowerblue', zorder=2)
        # ax.fill_between(x=(sztemporalinv-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
        ax.fill_between(x=conf_int_sztemporalinv, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
                        zorder=0)
        # ax.scatter(sztemporalinv[:-1], avg_responses, c='orange', zorder=4)
        ax.set_ylim([-2, 2.5])
        ax.invert_xaxis()
        ax.set_title(
            f'photostim responses vs. distance to sz wavefront (binned every {results.binned__time_vs_photostimresponses["bin_width_sec"]}sec)',
            wrap=True)
        ax.set_xlabel('time to sz inv (secs)')
        ax.set_ylabel(TargetsSzInvasionTemporal.photostim_responses_zscore_type)
        ax.margins(0)
        ax.axhline(0, ls='--', lw=1, color='black')

        pixels = [np.array(num2)] * 10
        ax2.imshow(pixels, cmap='Greys', vmin=-5, vmax=100, aspect=0.05)
        ax2.axis('off')
        if not 'fig' in kwargs and not 'axes' in kwargs:
            fig.tight_layout(pad=1)
            fig.show()

class TargetsSzInvasionTemporalResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__TargetsSzInvasionTemporal.pkl'

    def __init__(self):
        super().__init__()
        self.percentiles = None
        self.responses_sorted = None
        self.scale_percentile_times = None
        self.times_to_sz_sorted = None
        self.data_all = None
        self.data = None  # output of plot_responses_vs_time_to_seizure_SLMTargets_2ddensity
        self.range_of_sz_invasion_time: list = [-1.0, -1.0, -1.0]  # TODO need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion time stats calculated across all targets and all exps
        self.binned__time_vs_photostimresponses = {'bin_width_sec': None, 'sztemporal_bins': None,
                                                      'num_points_in_bin': None,
                                                      'avg_photostim_response_in_bin': None,
                                                      '95conf_int': None}

    @classmethod
    def load(cls):
        return pj.load_pkl(cls.SAVE_PATH)


REMAKE = False
if not os.path.exists(TargetsSzInvasionTemporalResults.SAVE_PATH) or REMAKE:
    results = TargetsSzInvasionTemporalResults()
    results.save_results()
else:
    results: TargetsSzInvasionTemporalResults = TargetsSzInvasionTemporalResults.load()


# UTILITY CLASSES
class ExpobjStrippedTimeDelaySzInvasion:
    def __init__(self, expobj):
        self.t_series_name = expobj.t_series_name
        self.seizure_lfp_onsets = expobj.seizure_lfp_onsets
        self.seziure_lfp_offsets = expobj.seizure_lfp_offsets
        self.raw_SLMTargets = expobj.raw_SLMTargets
        self.mean_raw_flu_trace = expobj.meanRawFluTrace
        from _utils_._lfp import LFP
        self.lfp_downsampled = LFP.downsampled_LFP(expobj=expobj)
        self.save(expobj=expobj)

    def __repr__(self):
        return f"ExpobjStrippedTimeDelaySzInvasion - {self.t_series_name}"

    def save(self, expobj):
        path = expobj.analysis_save_path + '/export/'
        os.makedirs(path, exist_ok=True)
        path += f"{expobj.prep}_{expobj.trial}_stripped.pkl"
        with open(f'{path}', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            print(f"|- Successfully saved stripped expobj: {expobj.prep}_{expobj.trial} to {path}")


#  RUN FOR TARGETS SZ INVASION TEMPORAL RESULTS
if __name__ == '__main__':
    pass

# %% ARCHIVE

### below should be almost all refactored to TargetsSzInvasionTemporal class


# # %% 3) COLLECT TIME DELAY TO SZ INVASION FOR EACH TARGET AT EACH PHOTOSTIM TIME
#
# # create a df containing delay to sz invasion for each target for each stim frame (dim: n_targets x n_stims)
# @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
# def collect_time_delay_sz_stims(self, **kwargs):
#     expobj: Post4ap = kwargs['expobj']
#     df = pd.DataFrame(columns=expobj.PhotostimResponsesSLMTargets.adata.var['stim_start_frame'],
#                       index=expobj.slmtargets_data.obs.index)
#     for target in expobj.slmtargets_data.obs.index:
#         # target = 0
#         print(f'\- collecting from target #{target} ... ') if (int(target) % 10) == 0 else None
#         cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
#         sz_times = expobj.slmtargets_data.obs.iloc[int(target), cols_]
#         fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if
#                     not pd.isnull(time)]
#         for szstart, szstop in zip(expobj.seizure_lfp_onsets, expobj.seizure_lfp_offsets):
#             fr_time = [i for i in fr_times if szstart < i < szstop]
#             if len(fr_time) >= 1:
#                 for stim in expobj.stim_start_frames:
#                     if szstart < stim < szstop:
#                         time_del = - (stim - fr_time[0]) / expobj.fps
#                         df.loc[target, stim] = round(time_del, 2)  # secs
#
#     self.time_del_szinv_stims = df
#     self._update_expobj(expobj)
#     expobj.save()
#
#
# # collect_time_delay_sz_stims()
#
# @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=True)
# def check_collect_time_delay_sz_stims(**kwargs):
#     expobj: Post4ap = kwargs['expobj']
#
#     # plot num stims outside of sz invasion and num stims after sz invasion
#     pos_values_collect = []
#     neg_values_collect = []
#     for idx, row in expobj.time_del_szinv_stims.iterrows():
#         values = row[row.notnull()]
#         pos_values_collect += list(values[values > 0])
#         neg_values_collect += list(values[values < 0])
#
#     expobj.time_delay_sz_stims_pos_values_collect = pos_values_collect
#     expobj.time_delay_sz_stims_neg_values_collect = neg_values_collect
#
#     print(f"avg num stim before sz invasion: {len(pos_values_collect) / expobj.n_targets_total}")
#     print(f"avg num stim after sz invasion: {len(neg_values_collect) / expobj.n_targets_total}")
#
#
# # check_collect_time_delay_sz_stims()
#
# @plot_piping_decorator(figsize=(8, 4), nrows=1, ncols=1, verbose=False)
# def plot_time_delay_sz_stims(**kwargs):
#     fig, ax, expobj = kwargs['fig'], kwargs['ax'], kwargs['expobj']
#
#     ax.hist(expobj.time_delay_sz_stims_pos_values_collect, fc='blue')
#     ax.hist(expobj.time_delay_sz_stims_neg_values_collect, fc='purple')
#
#
# # %% 1) collecting mean of seizure invasion Flu traces from all targets for each experiment
#
# @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
# def check_targets_sz_invasion_time(**kwargs):
#     expobj: Post4ap = kwargs['expobj']
#     print(expobj.slmtargets_data.obs)
#

# check_targets_sz_invasion_time()
#
#
# create dictionary containing mean Raw Flu trace around sz invasion time of each target (as well as other info as keyed into dict)
# @Utils.run_for_loop_across_exps(run_pre4ap_trials=False,
#                                 run_post4ap_trials=True, allow_rerun=1)  # , run_trials=['RL109 t-016', 'PS07 t-011', 'PS11 t-011'])
# def collect_targets_sz_invasion_traces(**kwargs):
#     expobj: Post4ap = kwargs['expobj']
#
#     fov_traces_ = []
#     traces_ = []
#     pre = 5
#     post = 10
#     for target, coord in enumerate(expobj.slmtargets_data.obs['SLM target coord']):
#         # target, coord = 0, expobj.slmtargets_data.obs['SLM target coord'][0]
#         print(f'\- collecting from target #{target} ... ') if (target % 10) == 0 else None
#         cols_ = [idx for idx, col in enumerate([*expobj.TargetsSzInvasionTemporal.adata.obs]) if 'time_del_sz' in col]
#         assert len(cols_) > 0
#         sz_times = expobj.TargetsSzInvasionTemporal.adata.obs.iloc[target, cols_]
#         fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if not pd.isnull(time)]
#
#         # collect each frame seizure invasion time Flu snippet for current target
#         target_traces = []
#         fov_traces = []
#         for fr in fr_times:
#             target_tr = expobj.raw_SLMTargets[target][fr - int(pre * expobj.fps): fr + int(post * expobj.fps)]
#             fov_tr = expobj.meanRawFluTrace[fr - int(pre * expobj.fps): fr + int(post * expobj.fps)]
#             target_traces.append(target_tr)
#             fov_traces.append(fov_tr)
#             # ax.plot(pj.moving_average(to_plot, n=4), alpha=0.2)
#         traces_.append(np.mean(target_traces, axis=0)) if len(target_traces) > 1 else None
#         fov_traces_.append(np.mean(fov_traces, axis=0)) if len(target_traces) > 1 else None
#
#     traces_ = np.array(traces_)
#     fov_traces_ = np.array(fov_traces_)
#     MEAN_TRACE = np.mean(traces_, axis=0)
#     FOV_MEAN_TRACE = np.mean(fov_traces_, axis=0)
#     assert len(FOV_MEAN_TRACE) > 1
#     expobj.mean_targets_szinvasion_trace = {'fov_mean_trace': FOV_MEAN_TRACE,
#                                             'mean_trace': MEAN_TRACE,
#                                             'pre_sec': pre,
#                                             'post_sec': post}
#     expobj.save()


#

# # %% 2) PLOT TARGETS FLU PERI- TIME TO INVASION
# #
# def plot_targets_sz_invasion_meantraces_full():
#     fig, ax = plt.subplots(figsize=[3, 6])
#
#     @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
#     def plot_targets_sz_invasion_meantraces(**kwargs):
#         expobj: Post4ap = kwargs['expobj']
#
#         fov_mean_trace, to_plot, pre, post = expobj.mean_targets_szinvasion_trace['fov_mean_trace'], \
#                                              expobj.mean_targets_szinvasion_trace['mean_trace'], \
#                                              expobj.mean_targets_szinvasion_trace['pre_sec'], \
#                                              expobj.mean_targets_szinvasion_trace['post_sec']
#         invasion_spot = int(pre * expobj.fps)
#         to_plot = pj.moving_average(to_plot, n=6)
#         fov_mean_trace = pj.moving_average(fov_mean_trace, n=6)
#         to_plot_normalize = (to_plot - to_plot[0]) / to_plot[0]
#         fov_mean_normalize = (fov_mean_trace - fov_mean_trace[0]) / fov_mean_trace[0]
#
#         x_time = np.linspace(-pre, post, len(to_plot))
#
#         ax.plot(x_time, to_plot_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3)
#         # ax2.plot(x_time, fov_mean_normalize, color=pj.make_random_color_array(n_colors=1)[0], linewidth=3, alpha=0.5, linestyle='--')
#
#         ax.scatter(x=0, y=to_plot_normalize[invasion_spot], color='crimson', s=45, zorder=5)
#
#         xticks = [-pre, 0, post]
#         # xticks_loc = [xtick*expobj.fps for xtick in [0, pre, pre+post]]
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticks)
#         ax.set_xlabel('Time (secs) to sz invasion')
#         ax.set_ylabel('Flu change (norm.)')
#         # ax.set_xlim([np.min(xticks_loc), np.max(xticks_loc)])
#         # ax.set_title(f"{expobj.t_series_name}")
#
#     plot_targets_sz_invasion_meantraces()
#     fig.suptitle('avg Flu at sz invasion', wrap=True, y=0.96)
#     fig.tight_layout(pad=2)
#     fig.show()
#
#
#
#
if __name__ == '__main__':
    plot_targets_sz_invasion_meantraces_full()
    collect_targets_sz_invasion_traces()

#     # collect_time_delay_sz_stims()
#     # check_collect_time_delay_sz_stims()
#       plot_time_delay_sz_stims()
#     #
#       check_targets_sz_invasion_time()
#     pass
