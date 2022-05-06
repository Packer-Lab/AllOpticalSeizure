## this file for code review for Targets Sz Invasion Spatial analysis
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os.path
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


"""
TODO:
- classify targets during sz as proximal or distal to sz wavefront
- shuffle distances of targets and replot distance vs. responses as a control comparison.
    
"""

class SLMTargets(Quantification):
    pass



run_trials=['RL108 t-013']
class TargetsSzInvasionSpatial_codereview(SLMTargets):
    response_type = 'dFF (zscored) (interictal)'

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\t\- ADDING NEW TargetsSzInvasionSpatial MODULE codereview to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionSpatial <-- Quantification Analysis submodule codereview for expobj <{self.expobj_id}>"

    # 1.0) calculate/collect min distance to seizure and responses at each distance ###############################

    def _create_anndata(self, expobj: Post4ap, distance_to_sz_array):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object primarily based around min distance to sz wavefront.

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
        _data_type = 'SLM Targets - distance to sz wavefront (pixels)'  # primary data label

        print(f"\t\----- CREATING annotated data object using AnnData:")
        # create anndata object
        from _utils_._anndata import AnnotatedData2
        distance_to_sz_array_adata = AnnotatedData2(X=distance_to_sz_array, obs=obs_meta, var=var_meta.T, data_label=_data_type)

        print(f"Created: {distance_to_sz_array_adata}")
        self.adata = distance_to_sz_array_adata
        # self.add_photostimresponses()

    # def add_photostimresponses(self, expobj):
    #     responses = expobj.PhotostimResponsesSLMTargets.adata.layers[self.response_type]
    #     self.adata.add_layer(layer_name=response_type, data=responses)
    #     print(self.adata)


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0)
    def run__add__min_distance_to_seizure(**kwargs):
        "Run calculation of min distance of SLM target to seizure wavefront for each target at each sz photostim"
        print(f"\t\- collecting responses vs. distance to seizure [5.0-1]")
        force_redo = True

        expobj: Post4ap = kwargs['expobj']
        expobj.sz_locations_stims() if not hasattr(expobj, 'stimsSzLocations') else None
        distance_to_sz_df = expobj.calcMinDistanceToSz_newer(analyse_cells='SLM Targets', show_debug_plot=False) if not hasattr(expobj, 'distance_to_sz') or force_redo else None  ## <<-- TODO main code line to code review and confirm. try to confirm correct calculation of distances using some plots??
        # expobj.save()

        # distance_to_sz_arr = np.array(distance_to_sz_df)
        distance_to_sz_arr = np.array(distance_to_sz_df['SLM Targets'])

        # Add .distance_to_sz attribute (from .calcMinDistanceToSz()) as anndata layer of expobj.PhotostimResponsesSLMTargets.adata
        expobj.TargetsSzInvasionSpatial_codereview._create_anndata(expobj=expobj, distance_to_sz_array=distance_to_sz_arr)
        expobj.save()


    def add_sz_distance_um_layer(self, expobj: Post4ap):
        arr = self.adata.X
        newarr = np.round(arr / expobj.pix_sz_x, 2)
        self.adata.add_layer(layer_name='distance to sz (um)', data=newarr)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0)
    def run__add_sz_distance_um_layer(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        expobj.TargetsSzInvasionSpatial_codereview.add_sz_distance_um_layer(expobj=expobj)
        expobj.save()


    def class_targets_proximal_distal(self):
        """
        Add layer to anndata of matrix that contains classification of targets as proximal (<100um), middle (100um < x < 200um) distal (>200um) to sz distance.

        """

        print(self.adata.layers.keys())

        distances = self.adata.layers['distance to sz (um)']
        newarr = np.full_like(distances, np.nan, dtype='<U10')


        newarr[distances < 0] = 'insz'
        newarr[distances > 200] = 'distal'
        newarr[np.where((distances < 100) & (distances > 0), True, False)] = 'proximal'
        newarr[np.where((distances < 200) & (distances > 100), True, False)] = 'middle'

        self.adata.add_layer(layer_name='outsz location', data=newarr)


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0)
    def plot_sz_boundary_location(**kwargs):
        expobj = kwargs['expobj']
        aoplot.plot_sz_boundary_location(expobj)


    def collect__responses_vs_distance_to_seizure_SLMTargets(self, expobj: Post4ap, response_type: str = None):
        """
        Main function to setup data matrix for photostim responses and distances to seizure wavefront for SLM targets.

        :param expobj:
        :param response_type: either 'dFF (zscored)' or 'dFF (zscored) (interictal)'
        :param kwargs: must contain expobj as arg key
        """
        print(f"\t\- collecting responses vs. distance to seizure [5.0-2]")
        # expobj = kwargs['expobj']

        # # uncomment if need to rerun for a particular expobj....but shouldn't really need to be doing so
        # if not hasattr(expobj, 'responses_SLMtargets_tracedFF'):
        #     expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
        #         expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)
        #     print(f'WARNING: {expobj.t_series_name} had to rerun .get_SLMTarget_responses_dff')

        response_type = self.response_type if response_type is None else response_type

        # (re-)make pandas dataframe
        df = pd.DataFrame(columns=['target_id', 'stim_id', 'inorout_sz', 'distance_to_sz', response_type])

        stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if
                    stim in [*expobj.slmtargets_szboundary_stim]]

        # z_scored = expobj.responses_SLMtargets_tracedFF  # initializing z_scored df
        z_scored = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.adata.layers[self.response_type])  # retrieving zscored responses
        for idx, target in enumerate(expobj.responses_SLMtargets_tracedFF.index):
            # ## z-scoring of SLM targets responses: -- this is not necessary (right??) if using .PhotostimResponsesSLMTargets.adata.layers['dFF zscored...'] - as this should already be zscored appropriately (.22/03/15)
            # if response_type == 'dFF (zscored)' or response_type == 'dFF (zscored) (interictal)':
            #     # set a different slice of stims for different variation of z scoring
            #     if response_type == 'dFF (zscored)':
            #         slice = expobj.responses_SLMtargets_tracedFF.columns  # (z scoring all stims all together from t-series)
            #     elif response_type == 'dFF (zscored) (interictal)':
            #         slice = expobj.stim_idx_outsz  # (z scoring all stims relative TO the interictal stims from t-series)
            #     __mean = np.mean(expobj.responses_SLMtargets_tracedFF.loc[target, slice])
            #     __std = np.std(expobj.responses_SLMtargets_tracedFF.loc[target, slice], ddof=1)
            #     # __mean = expobj.responses_SLMtargets_tracedFF.loc[target, :].mean()
            #     # __std = expobj.responses_SLMtargets_tracedFF.loc[target, :].std(ddof=1)
            #
            #     __responses = expobj.responses_SLMtargets_tracedFF.loc[target, :]
            #     z_scored.loc[target, :] = (__responses - __mean) / __std  # zscore dFF responses and add to zscored matrix

            for idx, stim in stim_ids:
                if target in expobj.slmtargets_szboundary_stim[stim]:
                    inorout_sz = 'in'
                else:
                    inorout_sz = 'out'
                # except KeyError:
                #     inorout_sz = np.nan
                #     print('break here!! debug if - else loop if erroring out..')

                # distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]
                distance_to_sz = self.adata.X[target, idx]  # distance in pixels

                if response_type == 'dFF':
                    # response = expobj.responses_SLMtargets_tracedFF.loc[target, idx]
                    response = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.X).loc[target, idx]
                elif response_type == 'dFF (zscored)' or response_type == 'dFF (zscored) (interictal)':
                    response = z_scored.loc[target, idx]  # z - scoring of SLM targets responses:
                else:
                    raise ValueError(
                        'response_type arg must be `dFF` or `dFF (zscored)` or `dFF (zscored) (interictal)`')

                df = pd.concat([df, pd.DataFrame({'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                     response_type: response}, index=[idx])])

                # df = df.append(
                #     {'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                #      response_type: response}, ignore_index=True)

        self.responses_vs_distance_to_seizure_SLMTargets = df

        # convert distances to sz boundary (in pixels) to microns
        self.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz_um'] = [round(i / expobj.pix_sz_x, 2) for i in self.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz']]

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
    def run__collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
        "code review - looks good to go .22/03/15"
        expobj = kwargs['expobj']
        expobj.TargetsSzInvasionSpatial_codereview.collect__responses_vs_distance_to_seizure_SLMTargets(expobj=expobj,
                                                                                                        response_type=TargetsSzInvasionSpatial_codereview.response_type)
        expobj.save()

    # 1.1) PLOT - collect and plot targets responses for stims vs. distance #######################################
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0)
    def plot_responses_vs_distance_to_seizure_SLMTargets(response_type=response_type, **kwargs):
        # response_type = 'dFF (zscored)'

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
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=0)
    def plot_collection_response_distance(response_type=response_type, **kwargs):
        print(f"\t|- plotting a collection of plots measuring responses vs. distance to seizure [5.1-2]")
        expobj = kwargs['expobj']
        # response_type = 'dFF (zscored)'
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

    # 2.0) binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses
    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=0)
    def retrieve__responses_vs_distance_to_seizure(response_type, positive_distances_only=False,
                                                        plot=False, **kwargs):
        """
        Collect photostim responses and distances to seizure wavefront.

        code review chunk here - done .22/03/15, things look pretty straightforward in this function.

        :param response_type:
        :param positive_distances_only:
        :param plot:
        :param kwargs:
        :return: array that is a stack of distance_to_sz and  for each target

        """

        expobj: Post4ap = kwargs['expobj']
        print(f"\t|- retrieving responses and distance to seizure for each target, stim, {expobj.t_series_name}")

        self: TargetsSzInvasionSpatial_codereview = expobj.TargetsSzInvasionSpatial_codereview

        data_expobj = np.array([[], []]).T
        for target in expobj.responses_SLMtargets_tracedFF.index:
            indexes = self.responses_vs_distance_to_seizure_SLMTargets[self.responses_vs_distance_to_seizure_SLMTargets['target_id'] == target].index
            responses = np.array(self.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, response_type])
            distance_to_sz = np.asarray(self.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, 'distance_to_sz_um'])
            # distance_to_sz_ = np.array(list(expobj.distance_to_sz['SLM Targets'].loc[target, :]))

            if positive_distances_only:
                distance_to_sz_pos = np.where(distance_to_sz > 0)[0]
                responses_posdistances = responses[distance_to_sz_pos]

                _data = np.array([distance_to_sz_pos, responses_posdistances]).T
            else:
                _data = np.array([distance_to_sz, responses]).T

            data_expobj = np.vstack((_data, data_expobj))  # stack of distance_to_sz and  for each target


        # make binned plot using hist2d function
        distances_to_sz = data_expobj[:, 0]
        bin_size = 20  # um
        # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
        bins_num = 40

        pj.plot_hist2d(data=data_expobj, bins=bins_num, y_label=response_type, title=expobj.t_series_name,
                       figsize=(4, 2), x_label='distance to seizure (um)',
                       y_lim=[-2, 2]) if plot else None

        return data_expobj

    # 2.1) PLOT - binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses - represent the distances in percentile space
    @staticmethod
    def convert__responses_szdistances_percentile_space(input_data):
        """
        Main function for converting matrix of responses vs. sz wavefront distances to percentile space.

        :param input_data: list of `data_expobj` outputs from plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(). each iteration represents one experiment.
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

        # convert distances to sz wavefront to percentile space
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


    def create__shuffled_pct_space_distances_list(self):
        # TODO create a shuffled percentile space list to use as control for statistical tests
        pass


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
        xticks = [10, 30, 42, 50, 70, 90]  # percentile space
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

        bin = 10
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

        xticks = [10, 30, 42, 50, 70, 90]  # percentile space
        ax.set_xticks(ticks=xticks)
        labels = [scale_percentile_distances[x_] for x_ in xticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel('distance to seizure (um)')

        fig.tight_layout(pad=2)
        plt.show()


    # 3) distnace vs. photostim responses - no percentile normalization of distances
    @staticmethod
    def collect__binned__distance_v_responses():
        """collect distance vs. respnses for distance bins"""
        bin_width = 20  # um
        bins = np.arange(0, 500, bin_width)  # 0 --> 500 um, split in XXum bins
        num = [0 for _ in range(len(bins))]  # num of datapoints in binned distances
        y = [0 for _ in range(len(bins))]  # avg responses at distance bin
        responses = [[] for _ in range(len(bins))]  # collect all responses at distance bin

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=False)
        def add_dist_responses(bins, num, y, responses, **kwargs):
            expobj = kwargs['expobj']
            szinvspatial = expobj.TargetsSzInvasionSpatial_codereview

            # print(num)
            # print(y)

            for _, row in szinvspatial.responses_vs_distance_to_seizure_SLMTargets.iterrows():
                dist = row['distance_to_sz_um']
                response = row[szinvspatial.response_type]
                for i, bin in enumerate(bins[:-1]):
                    if bins[i] < dist < (bins[i + 1]):
                        num[i] += 1
                        y[i] += response
                        responses[i].append(response)

            return num, y, responses

        func_collector = add_dist_responses(bins=bins, num=num, y=y, responses=responses)

        num, y, responses = func_collector[-1][0], func_collector[-1][1], func_collector[-1][2]

        distances = bins + bin_width/2  # offset distance values to the center of each bin - for plotting

        avg_responses = [np.mean(responses_) for responses_ in responses]

        # calculate 95% ci for avg responses
        import scipy.stats as stats
        conf_int = np.array(
            [stats.t.interval(alpha=0.95, df=len(responses_) - 1, loc=np.mean(responses_), scale=stats.sem(responses_))
             for responses_ in responses])

        return bin_width, distances, num, avg_responses, conf_int


    @staticmethod
    def plot__responses_v_distance_no_normalization(results):
        """plotting of binned responses over distance as a step function, with heatmap showing # of datapoints"""
        # distances_bins = results.binned__distance_vs_photostimresponses['distance_bins']
        distances = results.binned__distance_vs_photostimresponses['distance_bins']
        avg_responses = results.binned__distance_vs_photostimresponses['avg_photostim_response_in_bin']
        conf_int = results.binned__distance_vs_photostimresponses['95conf_int']
        num2 = results.binned__distance_vs_photostimresponses['num_points_in_bin']

        conf_int_distances = pj.flattenOnce([[distances[i], distances[i + 1]] for i in range(len(distances) - 1)])
        conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
        conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])

        fig, axs = plt.subplots(figsize=(6, 5), nrows=2, ncols=1)
        # ax.plot(distances[:-1], avg_responses, c='cornflowerblue', zorder=1)
        ax = axs[0]
        ax2 = axs[1]
        ax.step(distances, avg_responses, c='cornflowerblue', zorder=2)
        # ax.fill_between(x=(distances-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
        ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
                        zorder=0)
        # ax.scatter(distances[:-1], avg_responses, c='orange', zorder=4)
        ax.set_ylim([-0.5, 0.8])
        ax.set_title(
            f'photostim responses vs. distance to sz wavefront (binned every {results.binned__distance_vs_photostimresponses["bin_width_um"]}um)',
            wrap=True)
        ax.set_xlabel('distance to sz wavefront (um)')
        ax.set_ylabel(TargetsSzInvasionSpatial_codereview.response_type)
        ax.margins(0)

        pixels = [np.array(num2)] * 10
        ax2.imshow(pixels, cmap='Greys', vmin=-5, vmax=150, aspect=0.1)
        # ax.show()

        fig.tight_layout(pad=1)
        fig.show()


# %% RESULTS SAVING CLASS

class TargetsSzInvasionSpatialResults_codereview(Results):
    SAVE_PATH = SAVE_LOC + 'Results__TargetsSzInvasionSpatial_codereview.pkl'
    response_type = TargetsSzInvasionSpatial_codereview.response_type

    def __init__(self):
        super().__init__()

        self.range_of_sz_spatial_distance: List[float] = [-1.0, -1.0, -1.0]  # need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion distance stats calculated across all targets and all exps - maybe each seizure across all exps should be the 'n'?
        self.responses_vs_distance_to_seizure = None
        self.binned__distance_vs_photostimresponses = {'bin_width_um': None,
                                                       'distance_bins': None,  # list of distances over which data was binned
                                                       'num_points_in_bin': None,  # number of datapoints in each indiv bin
                                                       'avg_photostim_response_in_bin': None,  # avg photostim response within each bin
                                                       '95conf_int': None  # confidence interval around the avg photostim responses at each bin
                                                       }  #: photostim responses averaged over binned distances --> used to create step function plot.
        self.data_all = None
        self.percentiles = None
        self.responses_sorted = None
        self.distances_to_sz_sorted = None
        self.scale_percentile_distances = None

    @classmethod
    def load(cls):
        return pj.load_pkl(cls.SAVE_PATH)

# REMAKE = False
# if not os.path.exists(TargetsSzInvasionSpatialResults_codereview.SAVE_PATH) or REMAKE:
#     RESULTS = TargetsSzInvasionSpatialResults_codereview()
#     RESULTS.save_results()
# else:
#     RESULTS: TargetsSzInvasionSpatialResults_codereview = TargetsSzInvasionSpatialResults_codereview.load()


# %% running processing and analysis pipeline

# running processing and analysis pipeline

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__initTargetsSzInvasionSpatial(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj._parsePVMetadata()
    expobj.TargetsSzInvasionSpatial_codereview = TargetsSzInvasionSpatial_codereview(expobj=expobj)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=1)
def run__collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial_codereview.collect__responses_vs_distance_to_seizure_SLMTargets(expobj=expobj, response_type=TargetsSzInvasionSpatial_codereview.response_type)
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__class_targets_proximal_distal(**kwargs):
    expobj = kwargs['expobj']
    expobj.TargetsSzInvasionSpatial_codereview.class_targets_proximal_distal()
    expobj.save()

if __name__ == '__main__':
    main = TargetsSzInvasionSpatial_codereview
    results = TargetsSzInvasionSpatialResults_codereview.load()

    'Running updated code pipeline for just one experiment all the way thru.'
    # run__initTargetsSzInvasionSpatial()  # <- code review done
    #
    # main.run__add__min_distance_to_seizure()  # <- code review done
    #
    # main.run__collect_responses_vs_distance_to_seizure_SLMTargets()  # <- code review done

    main.run__add_sz_distance_um_layer()

    run__class_targets_proximal_distal()

    # main.plot_responses_vs_distance_to_seizure_SLMTargets()

    # main.plot_collection_response_distance()

    # results.responses_vs_distance_to_seizure = main.retrieve__responses_vs_distance_to_seizure(response_type=main.response_type, positive_distances_only=False, plot=False)  # <- code review done

    # results.save_results()

    # results.data_all, results.percentiles, results.responses_sorted, results.distances_to_sz_sorted, \
    #     results.scale_percentile_distances = main.convert__responses_szdistances_percentile_space(input_data=results.responses_vs_distance_to_seizure)  # <- code review done
    # results.save_results()

    # # not sure why there are nans in the .distances_to_sz_sorted and .percentiles and .data_all[:, 0], but need to remove them....
    # nan_list = np.isnan(list(results.data_all[:, 0]))
    # results.data_all = results.data_all[~nan_list, :]
    # nan_indexes = np.where(np.isnan(list(results.percentiles)))[0]
    # results.percentiles = [val for i, val in enumerate(results.percentiles) if i not in nan_indexes]
    # results.responses_sorted = [val for i, val in enumerate(results.responses_sorted) if i not in nan_indexes]
    # results.distances_to_sz_sorted = [val for i, val in enumerate(results.distances_to_sz_sorted) if i not in nan_indexes]
    #
    # main.plot_density_responses_szdistances(response_type=results.response_type, data_all=results.data_all,
    #                                         distances_to_sz_sorted=results.distances_to_sz_sorted, scale_percentile_distances=results.scale_percentile_distances)
    #
    # main.plot_lineplot_responses_pctszdistances(results.percentiles, results.responses_sorted, response_type=results.response_type,
    #                                             scale_percentile_distances=results.scale_percentile_distances)

