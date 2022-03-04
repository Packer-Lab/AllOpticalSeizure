import os.path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import _alloptical_utils as Utils
import funcsforprajay.funcs as pj

from _analysis_._utils import Quantification, Results
from _main_.Post4apMain import Post4ap
import _utils_.alloptical_plotting as aoplot
SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


class TargetsSzInvasionSpatial(Quantification):
    response_type = 'dFF (z scored) (interictal)'

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\t\- ADDING NEW TargetsSzInvasionSpatial MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionSpatial <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    ###### 1.0) calculate/collect min distance to seizure and responses at each distance ###############################

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0, skip_trials=['PS04 t-018'])
    def run_calculating_min_distance_to_seizure(**kwargs):
        print(f"\t\- collecting responses vs. distance to seizure [5.0-1]")

        expobj = kwargs['expobj']
        if not hasattr(expobj, 'stimsSzLocations'):
            expobj.sz_locations_stims()
        x_ = expobj.calcMinDistanceToSz()
        return x_
        # no_slmtargets_szboundary_stim.append(x_)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0, skip_trials=['PS04 t-018'])
    def plot_sz_boundary_location(**kwargs):
        expobj = kwargs['expobj']
        aoplot.plot_sz_boundary_location(expobj)

    def collect_responses_vs_distance_to_seizure_SLMTargets(self, response_type: str, expobj: Post4ap):
        """
    
        :param expobj:
        :param response_type: either 'dFF (z scored)' or 'dFF (z scored) (interictal)'
        :param kwargs: must contain expobj as arg key
        """
        print(f"\t\- collecting responses vs. distance to seizure [5.0-2]")
        # expobj = kwargs['expobj']

        # # uncomment if need to rerun for a particular expobj....but shouldn't really need to be doing so
        # if not hasattr(expobj, 'responses_SLMtargets_tracedFF'):
        #     expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
        #         expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)
        #     print(f'WARNING: {expobj.t_series_name} had to rerun .get_SLMTarget_responses_dff')

        # (re-)make pandas dataframe
        df = pd.DataFrame(columns=['target_id', 'stim_id', 'inorout_sz', 'distance_to_sz', response_type])

        stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if
                    stim in expobj.distance_to_sz['SLM Targets'].columns]
        for idx, target in enumerate(expobj.responses_SLMtargets_tracedFF.index):

            ## z-scoring of SLM targets responses:
            z_scored = expobj.responses_SLMtargets_tracedFF  # initializing z_scored df
            if response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)':
                # set a different slice of stims for different variation of z scoring
                if response_type == 'dFF (z scored)':
                    slice = expobj.responses_SLMtargets_tracedFF.columns  # (z scoring all stims all together from t-series)
                elif response_type == 'dFF (z scored) (interictal)':
                    slice = expobj.stim_idx_outsz  # (z scoring all stims relative TO the interictal stims from t-series)
                __mean = expobj.responses_SLMtargets_tracedFF.loc[target, slice].mean()
                __std = expobj.responses_SLMtargets_tracedFF.loc[target, slice].std(ddof=1)
                # __mean = expobj.responses_SLMtargets_tracedFF.loc[target, :].mean()
                # __std = expobj.responses_SLMtargets_tracedFF.loc[target, :].std(ddof=1)

                __responses = expobj.responses_SLMtargets_tracedFF.loc[target, :]
                z_scored.loc[target, :] = (__responses - __mean) / __std

            for idx, stim in stim_ids:
                if target in expobj.slmtargets_szboundary_stim[stim]:
                    inorout_sz = 'in'
                else:
                    inorout_sz = 'out'

                distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]

                if response_type == 'dFF':
                    response = expobj.responses_SLMtargets_tracedFF.loc[target, idx]
                elif response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)':
                    response = z_scored.loc[target, idx]  # z - scoring of SLM targets responses:
                else:
                    raise ValueError(
                        'response_type arg must be `dFF` or `dFF (z scored)` or `dFF (z scored) (interictal)`')

                df = pd.concat([df, pd.DataFrame({'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                     response_type: response}, index=[idx])])

                # df = df.append(
                #     {'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                #      response_type: response}, ignore_index=True)

        self.responses_vs_distance_to_seizure_SLMTargets = df

        # convert distances to microns
        self.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz_um'] = [
            round(i / expobj.pix_sz_x, 2) for i in expobj.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz']]

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0, skip_trials=['PS04 t-018'])
    def run__collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
        expobj = kwargs['expobj']
        expobj.TargetsSzInvasionSpatial.collect_responses_vs_distance_to_seizure_SLMTargets(expobj=expobj,
                                                                                            response_type=TargetsSzInvasionSpatial.response_type)
        expobj.save()

    # TODO need to review below (everything above shouuuuld be working)
    ###### 1.1) PLOT - collect and plot targets responses for stims vs. distance #######################################
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0, skip_trials=['PS04 t-018'])
    def plot_responses_vs_distance_to_seizure_SLMTargets(response_type=response_type, **kwargs):
        # response_type = 'dFF (z scored)'

        print(f"\t|- plotting responses vs. distance to seizure [5.1-1]")
        expobj = kwargs['expobj']
        fig, ax = plt.subplots(figsize=[8, 3])
        for target in expobj.responses_SLMtargets_tracedFF.index:
            idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if
                               stim in expobj.distance_to_sz['SLM Targets'].columns]

            stim_sz_boundary = [stim for idx, stim in enumerate(expobj.stim_start_frames) if
                               stim in expobj.distance_to_sz['SLM Targets'].columns]

            responses = np.array(expobj.responses_SLMtargets_tracedFF.loc[target, idx_sz_boundary])
            distance_to_sz = np.array(expobj.distance_to_sz['SLM Targets'].loc[target, stim_sz_boundary])

            positive_distances = np.where(distance_to_sz > 0)
            negative_distances = np.where(distance_to_sz < 0)

            pj.make_general_scatter(x_list=[distance_to_sz[positive_distances]], y_data=[responses[positive_distances]],
                                    fig=fig, ax=ax, colors=['cornflowerblue'], alpha=0.5, s=30, show=False,
                                    x_label='distance to sz', y_label=response_type)
            pj.make_general_scatter(x_list=[distance_to_sz[negative_distances]], y_data=[responses[negative_distances]],
                                    fig=fig, ax=ax, colors=['tomato'], alpha=0.5, s=30, show=False,
                                    x_label='distance to sz', y_label=response_type)

        fig.suptitle(expobj.t_series_name)
        fig.show()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0, skip_trials=['PS04 t-018'])
    def plot_collection_response_distance(response_type=response_type, **kwargs):
        print(f"\t|- plotting a collection of plots measuring responses vs. distance to seizure [5.1-2]")
        expobj = kwargs['expobj']
        # response_type = 'dFF (z scored)'
        if not hasattr(expobj, 'responses_SLMtargets_tracedFF_avg_df'):
            expobj.avgResponseSzStims_SLMtargets(save=True)

        data = expobj.responses_vs_distance_to_seizure_SLMTargets
        fig, axs = plt.subplots(ncols=5, nrows=1, figsize=[18, 4])
        axs[0] = sns.boxplot(data=expobj.responses_SLMtargets_tracedFF_avg_df, x='stim_group', y='avg targets response',
                             order=['interictal', 'ictal'],
                             width=0.5, ax=axs[0],
                             palette=['tomato', 'mediumseagreen'])  # plotting mean across stims (len= # of targets)
        axs[0] = sns.swarmplot(data=expobj.responses_SLMtargets_tracedFF_avg_df, x='stim_group',
                               y='avg targets response', order=['interictal', 'ictal'],
                               color=".25", ax=axs[0])
        sns.stripplot(x="inorout_sz", y="distance_to_sz_um", data=data, ax=axs[1], alpha=0.2, order=['in', 'out'])
        axs[2] = sns.violinplot(x="inorout_sz", y=response_type, data=data, legend=False, ax=axs[2],
                                order=['in', 'out'])
        axs[2].set_ylim([-3, 3])
        axs[3] = sns.scatterplot(data=data, x='distance_to_sz_um', y=response_type, ax=axs[3], alpha=0.2,
                                 hue='distance_to_sz_um', hue_norm=(-1, 1),
                                 palette=sns.diverging_palette(240, 10, as_cmap=True), legend=False)
        axs[3].set_ylim([-3, 3])
        aoplot.plot_sz_boundary_location(expobj, fig=fig, ax=axs[4], title=None)
        fig.suptitle(f"{expobj.t_series_name} - {response_type}")
        fig.tight_layout(pad=1.1)
        fig.show()

    ###### 2.0) PLOT - binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0, skip_trials=['PS04 t-018'])
    def plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(response_type, positive_distances_only=False,
                                                                   plot=True, **kwargs):

        print(f"\t|- binning responses vs. distance to seizure")
        expobj = kwargs['expobj']

        data_expobj = np.array([[], []]).T
        for target in expobj.responses_SLMtargets_tracedFF.index:
            indexes = expobj.responses_vs_distance_to_seizure_SLMTargets[
                expobj.responses_vs_distance_to_seizure_SLMTargets['target_id'] == target].index
            responses = np.array(expobj.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, response_type])
            distance_to_sz = np.asarray(
                expobj.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, 'distance_to_sz_um'])
            # distance_to_sz_ = np.array(list(expobj.distance_to_sz['SLM Targets'].loc[target, :]))

            if positive_distances_only:
                distance_to_sz_pos = np.where(distance_to_sz > 0)[0]
                responses_posdistances = responses[distance_to_sz_pos]

                _data = np.array([distance_to_sz_pos, responses_posdistances]).T
            else:
                _data = np.array([distance_to_sz, responses]).T

            data_expobj = np.vstack((_data, data_expobj))

        distances_to_sz = data_expobj[:, 0]
        bin_size = 20  # um
        # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
        bins_num = 40

        pj.plot_hist2d(data=data_expobj, bins=bins_num, y_label=response_type, title=expobj.t_series_name,
                       figsize=(4, 2), x_label='distance to seizure (um)',
                       y_lim=[-2, 2]) if plot else None

        return data_expobj

    ###### 2.1) PLOT - binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses - represent the distances in percentile space
    @staticmethod
    def convert_responses_szdistances_percentile_space(input_data):
        """
        convert matrix of responses vs. sz wavefront distances to percentile space.

        :param input_data: list of `data_expobj` outputs from plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity().
        :return:
        """

        print(f"Converting matrix of responses vs. sz wavefront distances to percentile space.")

        data_all = np.array([[], []]).T
        for idx, data_ in enumerate(input_data):
            data_all = np.vstack((data_, data_all))

        from scipy.stats import percentileofscore

        distances_to_sz = data_all[:, 0]
        idx_sorted = np.argsort(distances_to_sz)
        distances_to_sz_sorted = distances_to_sz[idx_sorted]
        responses_sorted = data_all[:, 1][idx_sorted]
        # s = pd.Series(distances_to_sz_sorted)
        print(f"\n\tconverting distances to percentiles {len(distances_to_sz_sorted)}...")

        percentiles = []
        for idx, x in enumerate(distances_to_sz_sorted):
            print(f"\n\tprogress at {idx} out of {len(distances_to_sz_sorted)}") if idx % 1000 == 0 else None
            percentiles.append(percentileofscore(distances_to_sz_sorted, x))


        # percentiles = [percentileofscore(distances_to_sz_sorted, x) for x in distances_to_sz_sorted]

        # from numba.typed import List
        # from numba import njit
        #
        # @njit
        # def convert_percentile(list_scores):
        #     percentiles = []
        #
        #     for x in list_scores:
        #         percentiles.append(percentileofscore(list_scores, x))
        #
        #     return percentiles
        #
        # percentiles = convert_percentile(distances_to_sz_sorted)

        # percentiles = s.apply(lambda x: percentileofscore(distances_to_sz_sorted, x))

        # from numba.typed import Dict
        # from numba import njit

        scale_percentile_distances = {}
        for pct in range(0, 100):
            print(f"\n\tprogress at {pct} percentile") if pct % 5 == 0 else None
            scale_percentile_distances[int(pct + 1)] = np.round(np.percentile(distances_to_sz_sorted, pct), 0)


        data_all = np.array([percentiles, responses_sorted]).T

        return data_all, percentiles, responses_sorted, distances_to_sz_sorted, scale_percentile_distances

    @staticmethod
    def plot_density_responses_szdistances(response_type, data_all, distances_to_sz_sorted, scale_percentile_distances):
        # plotting density plot for all exps, in percentile space (to normalize for excess of data at distances which are closer to zero) - TODO any smoothing?

        bin_size = 20  # um
        # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
        bins_num = [100, 500]

        fig, ax = plt.subplots(figsize=(6, 3))
        pj.plot_hist2d(data=data_all, bins=bins_num, y_label=response_type, figsize=(6, 3),
                       x_label='distance to seizure (%tile space)',
                       title=f"2d density plot, all exps, 50%tile = {np.percentile(distances_to_sz_sorted, 50)}um",
                       y_lim=[-3, 3], fig=fig, ax=ax, show=False)
        ax.axhline(0, ls='--', c='white', lw=1)
        xticks = [1, 25, 50, 57, 75, 100]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_distances[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel('distance to seizure (um)')

        fig.show()

    # plotting line plot for all datapoints for responses vs. distance to seizure
    @staticmethod
    def plot_lineplot_responses_pctszdistances(percentiles, responses_sorted, response_type,
                                               scale_percentile_distances):
        percentiles_binned = np.round(percentiles)

        bin = 20
        # change to pct% binning
        percentiles_binned = (percentiles_binned // bin) * bin

        d = {'distance to seizure (%tile space)': percentiles_binned,
             response_type: responses_sorted}

        df = pd.DataFrame(d)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=df, x='distance to seizure (%tile space)', y=response_type, ax=ax)
        ax.set_title(f'responses over distance to sz, all exps, normalized to percentile space ({bin}% bins)',
                     wrap=True)
        ax.margins(0.02)
        ax.axhline(0, ls='--', c='orange', lw=1)

        xticks = [1, 25, 50, 57, 75, 100]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_distances[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel('distance to seizure (um)')

        fig.tight_layout(pad=2)
        plt.show()


@dataclass
class TargetsSzInvasionSpatialResults(Results):
    save_path = SAVE_LOC + 'Results__TargetsSzInvasionSpatial.pkl'
    response_type = TargetsSzInvasionSpatial.response_type

    range_of_sz_spatial_distance: List[
        float] = None  # need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion distance stats calculated across all targets and all exps - maybe each seizure across all exps should be the 'n'?

    no_slmtargets_szboundary_stim = []
    data_all = None
    percentiles = None
    responses_sorted = None
    distances_to_sz_sorted = None
    scale_percentile_distances = None

    @classmethod
    def load(cls):
        return pj.load_pkl(cls.save_path)



# try:
#     Results__TargetsSzInvasionSpatial = pj.load_pkl(TargetsSzInvasionSpatialResults.save_path)
# except FileNotFoundError:

if not os.path.exists(TargetsSzInvasionSpatialResults.save_path):
    Results__TargetsSzInvasionSpatial = TargetsSzInvasionSpatialResults()
    Results__TargetsSzInvasionSpatial.save_results()



# %% running processing and analysis pipeline

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0, skip_trials=['PS04 t-018'])
def run__initTargetsSzInvasionSpatial(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial = TargetsSzInvasionSpatial(expobj=expobj)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0, skip_trials=['PS04 t-018'])
def run__collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial.collect_responses_vs_distance_to_seizure_SLMTargets(expobj=expobj, response_type=TargetsSzInvasionSpatial.response_type)
    expobj.save()


# %%
if __name__ == '__main__':
    # run__initTargetsSzInvasionSpatial()
    # TargetsSzInvasionSpatialResults.no_slmtargets_szboundary_stim = TargetsSzInvasionSpatial.run_calculating_min_distance_to_seizure()
    #
    # TargetsSzInvasionSpatial.run__collect_responses_vs_distance_to_seizure_SLMTargets()
    #
    # TargetsSzInvasionSpatialResults.save_results()
    #
    # TODO need to review below (the code runs above shouuuuld be working)

    # TargetsSzInvasionSpatial.plot_responses_vs_distance_to_seizure_SLMTargets()
    #
    # TargetsSzInvasionSpatial.plot_collection_response_distance()
    #
    # TargetsSzInvasionSpatialResults.data = TargetsSzInvasionSpatial.plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(response_type=TargetsSzInvasionSpatial.response_type, positive_distances_only=False, plot=False)

    Results__TargetsSzInvasionSpatial = TargetsSzInvasionSpatialResults.load()

    # Results__TargetsSzInvasionSpatial.data_all, Results__TargetsSzInvasionSpatial.percentiles, Results__TargetsSzInvasionSpatial.responses_sorted, \
    #     Results__TargetsSzInvasionSpatial.distances_to_sz_sorted, Results__TargetsSzInvasionSpatial.scale_percentile_distances = TargetsSzInvasionSpatial.convert_responses_szdistances_percentile_space(input_data=Results__TargetsSzInvasionSpatial.data)
    #
    # Results__TargetsSzInvasionSpatial.save_results()

    TargetsSzInvasionSpatial.plot_density_responses_szdistances(response_type=Results__TargetsSzInvasionSpatial.response_type,
                                                                data_all=Results__TargetsSzInvasionSpatial.data_all,
                                                                distances_to_sz_sorted=Results__TargetsSzInvasionSpatial.distances_to_sz_sorted,
                                                                scale_percentile_distances=Results__TargetsSzInvasionSpatial.scale_percentile_distances)
    TargetsSzInvasionSpatial.plot_lineplot_responses_pctszdistances(Results__TargetsSzInvasionSpatial.percentiles,
                                                                    Results__TargetsSzInvasionSpatial.responses_sorted,
                                                                    response_type=Results__TargetsSzInvasionSpatial.response_type,
                                                                    scale_percentile_distances=Results__TargetsSzInvasionSpatial.scale_percentile_distances)

