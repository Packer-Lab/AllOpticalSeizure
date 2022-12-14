import sys

from funcsforprajay.funcs import calc_distance_2points
from matplotlib import pyplot as plt

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os
from typing import Union, List

import numpy as np
import pandas as pd
from scipy import stats

import _alloptical_utils as Utils
from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot
from funcsforprajay import funcs as pj

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-firing_rates-sz_distance-Non-targets/"


# %% ###### NON TARGETS analysis + plottings

class NonTargetsSzInvasionSpatialResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesNonTargets.pkl'

    def __init__(self):
        super().__init__()

        self.summed_responses = None  #: dictionary of baseline and interictal summed responses of targets and nontargets across experiments
        self.lin_reg_summed_responses = None  #: dictionary of baseline and interictal linear regression metrics for relating total targets responses and nontargets responses across experiments
        self.avg_responders_num = None  #: average num responders, for pairedmatched experiments between baseline pre4ap and interictal
        self.avg_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal
        self.rolling_binned__distance_vs_photostimresponses = None  #: distance to seizure boundary vs. firing rates for rolling distance bins for nontargets

REMAKE = False
if not os.path.exists(NonTargetsSzInvasionSpatialResults.SAVE_PATH) or REMAKE:
    results = NonTargetsSzInvasionSpatialResults()
    results.save_results()


class NonTargetsSzInvasionSpatial(Quantification):
    """class for classifying nontargets relative to sz boundary and quantifying distance to sz boundary.

    Tasks:
    [x] code for splitting nontargets inside and outside sz boundary - during ictal stims
    [x] run through for all experiments - transfer over from the NontargetsPhotostimResponsesQuant class

    """

    save_path = SAVE_LOC + 'NonTargetsSzInvasionSpatial.pkl'
    EXCLUDE_TRIALS = ['PS04 t-012',  # no responding cells and also doubled up by PS04 t-017 in any case
                      ]

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)

        self._classify_nontargets_szboundary(expobj=expobj, force_redo=False)
        # expobj.save()
        distance_to_sz_um = self._calc_min_distance_sz_nontargets(expobj=expobj)
        self.adata: AnnotatedData2 = self.create_anndata(expobj=expobj,
                                                         distance_to_sz=distance_to_sz_um)  #: anndata table to hold data about distances to seizure boundary for nontargets
        self._add_nontargets_sz_boundary_anndata()
        print(f'\- ADDING NEW NonTargetsSzInvasionSpatial MODULE to expobj: {expobj.t_series_name}')

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1,
                                    skip_trials=EXCLUDE_TRIALS)
    def run__NonTargetsSzInvasionSpatial(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.NonTargetsSzInvasionSpatial = NonTargetsSzInvasionSpatial(expobj=expobj)
        expobj.save()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1, skip_trials=EXCLUDE_TRIALS, set_cache=False)
    def run__methods(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        # distances_um = expobj.NonTargetsSzInvasionSpatial._calculate_distance_to_target(expobj=expobj)
        # expobj.NonTargetsSzInvasionSpatial._add_nontargets_distance_to_targets_anndata(expobj.PhotostimResponsesNonTargets.adata, distances_um)
        # expobj.NonTargetsSzInvasionSpatial._add_nontargets_idx_anndata()
        expobj.NonTargetsSzInvasionSpatial.calculate_firing_rate_sz_boundary(expobj=expobj)
        expobj.save()

    def __repr__(self):
        return f"NonTargetsSzInvasionSpatial <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    # 0) CLASSIFY NONTARGETS ACROSS SEIZURE BOUNDARY, INCLUDING MEASURING DISTANCE TO SEIZURE BOUNDARY #################
    @staticmethod
    def _classify_nontargets_szboundary(expobj: Post4ap, force_redo=False):
        """run procedure for classifying sz boundary with nontargets"""

        if not hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim') or force_redo:
            expobj.ExpSeizure._procedure__classifying_sz_boundary(expobj=expobj, cells='nontargets')

        assert hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim')
        if not hasattr(expobj, 'stimsSzLocations'): expobj.sz_locations_stims()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_post4ap_trials=False, run_trials=['RL109 t-018', 'PS06 t-013'], allow_rerun=1)
    def run__classify_nontargets_szboundary(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        expobj.NonTargetsSzInvasionSpatial._classify_nontargets_szboundary(expobj=expobj, force_redo=True)
        expobj.save()

    def _calc_min_distance_sz_nontargets(self, expobj: Post4ap):
        assert hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim')

        distance_to_sz_df = expobj.calcMinDistanceToSz_newer(analyse_cells='s2p nontargets', show_debug_plot=False)

        assert distance_to_sz_df.shape == expobj.PhotostimResponsesNonTargets.adata.shape

        return distance_to_sz_df / expobj.pix_sz_x

    # 1) CREATE ANNDATA - using distance to sz dataframe collected above ###############################################
    def create_anndata(self, expobj: Union[alloptical, Post4ap], distance_to_sz):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the photostim resposnes of all non-target ROIs.

        """

        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from non targets meta information

        obs_meta = pd.DataFrame(
            columns=['original_index', 'footprint', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'radius',
                     'aspect_ratio', 'npix_norm', 'skew', 'std'], index=distance_to_sz.index)
        for i, idx in enumerate(obs_meta.index):
            assert idx not in expobj.s2p_nontargets_exclude
            _stat = expobj.stat[expobj.cell_id.index(idx)]
            for __column in obs_meta:
                obs_meta.loc[idx, __column] = _stat[__column]

        # build numpy array for multidimensional obs metadata
        obs_m = {'ypix': [],
                 'xpix': []}
        for col in [*obs_m]:
            for i, idx in enumerate(expobj.s2p_nontargets):
                if idx not in expobj.s2p_nontargets_exclude:
                    obs_m[col].append(expobj.stat[i][col])
            obs_m[col] = np.asarray(obs_m[col])

        var_meta = pd.DataFrame(
            index=['stim_group', 'im_time_secs', 'stim_start_frame', 'wvfront in sz', 'seizure location'],
            columns=range(len(expobj.stim_start_frames)))
        for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
            if 'pre' in expobj.exptype:
                var_meta.loc['wvfront in sz', fr_idx] = None
                var_meta.loc['seizure location', fr_idx] = None
                var_meta.loc['stim_group', fr_idx] = 'baseline'
            elif 'post' in expobj.exptype:
                if stim_frame in expobj.stimsWithSzWavefront:
                    var_meta.loc['wvfront in sz', fr_idx] = True
                    var_meta.loc['seizure location', fr_idx] = (
                        expobj.stimsSzLocations.coord1[stim_frame], expobj.stimsSzLocations.coord2[stim_frame])
                else:
                    var_meta.loc['wvfront in sz', fr_idx] = False
                    var_meta.loc['seizure location', fr_idx] = None
                var_meta.loc['stim_group', fr_idx] = 'ictal' if fr_idx in expobj.stims_in_sz else 'interictal'
            var_meta.loc['stim_start_frame', fr_idx] = stim_frame
            var_meta.loc['im_time_secs', fr_idx] = stim_frame / expobj.fps

        # SET PRIMARY DATA
        print(f"\t\----- CREATING annotated data object using AnnData:")
        # create anndata object
        distance_to_sz.index = distance_to_sz.index.astype(str)
        distance_to_sz.columns = var_meta.columns
        photostim_responses_adata = AnnotatedData2(X=distance_to_sz, obs=obs_meta, var=var_meta.T, obsm=obs_m,
                                                   data_label='distance to sz (um)')

        print(f"Created: {photostim_responses_adata}")
        return photostim_responses_adata

    def _add_nontargets_sz_boundary_anndata(self):
        """add layer to anndata table that splits nontarget cell in or out of sz boundary.
        1: outside of sz boundary
        0: inside of sz boundary

        """

        arr = np.full_like(self.adata.X, np.nan)

        arr[self.adata.X > 0] = 1
        arr[self.adata.X < 0] = 0

        self.adata.add_layer(layer_name='in/out sz', data=arr)

    def _add_nontargets_idx_anndata(self):
        """add index of nontarget from the original suite2p dataset into the the anndata table. this is to account for cells that have been removed from the analysis.
        """

        nontargets = list(self.adata.obs['original_index'])
        idx = [nontargets.index(cell) for cell in nontargets]
        assert len(idx) == self.adata.shape[0], 'incorrect number of cells retrieved'

        self.adata.add_observation(obs_name='s2p_data_idx', values=idx)

    def _add_nontargets_proximal_distal_sz_anndata(self):
        """
        Add layer to anndata of matrix that contains classification of targets as proximal (<100um), middle (100um < x < 200um) distal (>200um) to sz distance.

        """

        arr = np.full_like(self.adata.X, np.nan, dtype='<U10')

        arr[self.adata.X < 0] = 'insz'
        arr[self.adata.X > 200] = 'distal'
        arr[np.where((self.adata.X < 100) & (self.adata.X > 0), True, False)] = 'proximal'
        arr[np.where((self.adata.X < 200) & (self.adata.X > 100), True, False)] = 'middle'

        self.adata.add_layer(layer_name='outsz location', data=arr)

    # 2) measure firing rate of non targets relative to the distance from seizure boundary
    # 2.1)
    def calculate_firing_rate_sz_boundary(self, expobj: Post4ap):
        """
        Measure the suite2p derived  deconvolved firing rates for nontargets individually at each photostim where the sz boundary is defined.
        finally add as a layer to the anndata table.

        Includes a firing rate z-score to the interictal period.
        IPR:
            Adding suite2p neuropil signal collected in a similar manner as FR above.
            Adding suite2p raw_gcamp signal collected in a similar manner as FR above.

        :return:
        """
        print(self.adata)
        print(expobj.t_series_name)

        pre_stim_period = 0.25  # seconds
        frames = expobj.getFrames(seconds=pre_stim_period)
        fr_stims = np.zeros_like(self.adata.X)
        fr_stims_zscored = np.zeros_like(self.adata.X)

        neuropil = np.zeros_like(self.adata.X)
        neuropil_zscored = np.zeros_like(self.adata.X)

        raw_gcamp = np.zeros_like(self.adata.X)
        raw_gcamp_zscored = np.zeros_like(self.adata.X)



        interictal_frames = [frame for frame in range(expobj.n_frames) if frame not in expobj.seizure_frames]
        # list of interictal average deconvolved firing rates for each nontarget cell
        interictal_FR_mean = np.array([np.average(expobj.spks[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])
        interictal_FR_std = np.array([np.std(expobj.spks[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])

        interictal_neuropil_mean = np.array([np.average(expobj.neuropil[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])
        interictal_neuropil_std = np.array([np.std(expobj.neuropil[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])

        interictal_gcamp_mean = np.array([np.average(expobj.raw[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])
        interictal_gcamp_std = np.array([np.std(expobj.raw[cellidx, interictal_frames]) for cellidx in list(self.adata.obs['s2p_data_idx'])])


        for i, stim in enumerate(list(self.adata.var['stim_start_frame'])):
            for j in list(self.adata.obs['s2p_data_idx']):
                _frames = np.arange(stim - frames, stim).astype(int)

                fr = np.average(expobj.spks[j, _frames])
                fr_stims[j, i] = fr

                npil = np.average(expobj.neuropil[j, _frames])
                neuropil[j, i] = npil

                gc = np.average(expobj.raw[j, _frames])
                raw_gcamp[j, i] = gc


                # zscored relative to interictal periods
                zscore = pj.zscore(dat=fr, mean=interictal_FR_mean[j], std=interictal_FR_std[j])
                fr_stims_zscored[j, i] = zscore

                zscore = pj.zscore(dat=npil, mean=interictal_neuropil_mean[j], std=interictal_neuropil_std[j])
                neuropil_zscored[j, i] = zscore

                zscore = pj.zscore(dat=gc, mean=interictal_gcamp_mean[j], std=interictal_gcamp_std[j])
                raw_gcamp_zscored[j, i] = zscore


        self.adata.add_layer(layer_name='firing_rate_prestim', data=fr_stims)
        self.adata.add_layer(layer_name='firing_rate_prestim_zscored', data=fr_stims_zscored)

        self.adata.add_layer(layer_name='neuropil_prestim', data=neuropil)
        self.adata.add_layer(layer_name='neuropil_prestim_zscored', data=neuropil_zscored)

        self.adata.add_layer(layer_name='raw_gcamp_prestim', data=raw_gcamp)
        self.adata.add_layer(layer_name='raw_gcamp_prestim_zscored', data=raw_gcamp_zscored)

    # 2.2)
    def retrieve__FR_vs_distance_to_seizure(self, positive_distances_only=False, plot=False):
        """
        -- not really doing any
        Collect firing rates and distances to seizure wavefront.


        :param response_type:
        :param positive_distances_only:
        :param plot:
        :param kwargs:
        :return: array that is a stack of distance_to_sz and  for each target

        """

        # expobj: Post4ap = kwargs['expobj']
        # print(f"\t|- retrieving firing rates and distances to seizure for each non target cell: {expobj.t_series_name}")

        type = 'gcamp'

        if type == 'raw':
            firing_rate_mtx = self.adata.layers['firing_rate_prestim']
        elif type == 'zscored':
            firing_rate_mtx = self.adata.layers['firing_rate_prestim_zscored']
        elif type == 'gcamp':
            firing_rate_mtx = self.adata.layers['raw_gcamp_prestim_zscored']
        elif type == 'neuropil':
            firing_rate_mtx = self.adata.layers['neuropil_prestim_zscored']
        else:
            raise ValueError('type must be `raw` or `zscored`')

        data_expobj = np.array([[], []]).T
        for cell in self.adata.obs['s2p_data_idx']:
            firing_rates = np.asarray(firing_rate_mtx[cell, :])
            distance_to_sz = np.asarray(self.adata.X[cell, :])

            if positive_distances_only:
                __idx_pos = np.where(distance_to_sz > 0)[0]
                _data = np.array([distance_to_sz[__idx_pos], firing_rates[__idx_pos]]).T
            else:
                __idx = ~np.isnan(distance_to_sz)
                _data = np.array([distance_to_sz[__idx], firing_rates[__idx]]).T

            data_expobj = np.vstack((_data, data_expobj))  # stack of distance_to_sz and  for each target

        # make binned plot using hist2d function
        distances_to_sz = data_expobj[:, 0]
        bin_size = 20  # um
        # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
        bins_num = 40

        if plot:
            pj.plot_hist2d(data=data_expobj, bins=[bins_num], y_label='Firing rate', title=self.expobj_id, figsize=(4, 2), x_label='distance to seizure (um)')

        return data_expobj


    # 2.3) COLLECT - distance vs. firing rates - no percentile normalization of distances - ROLLING BINS
    @staticmethod
    def collect__binned__distance_v_firing_rates_rolling_bins(results, rerun=0):
        """collect distance to seizure boundary vs. firing rates for rolling distance bins for nontargets"""

        if not rerun and hasattr(results, 'rolling_binned__distance_vs_firingrates'): return

        type_fr = 'spks - zscored'
        results.rolling_binned__distance_vs_firingrates = {}
        for type_fr in ['spks - raw', 'spks - zscored', 'gcamp - zscored', 'neuropil - zscored']:

            print(f"\n\n\n{'-' * 10} RUNNING {type_fr} {'.' * 50}")

            distances = np.arange(0, 500)  # distance metric at each rolling bin start
            bin_width = 20  # um
            rolling_bins = [(i, i + 20) for i in range(0, 500)]
            # bins = np.arange(0, 500, bin_width)  # 0 --> 500 um, split in XXum bins
            num = [0 for _ in range(len(rolling_bins))][:-20]  # num of datapoints in binned distances
            y = [0 for _ in range(len(rolling_bins))][:-20]  # avg responses at distance bin
            firing_rates = [[] for _ in range(len(rolling_bins))][:-20]  # collect all responses at distance bin

            @Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, set_cache=False)
            def add_dist_responses(rolling_bins, num, y, firing_rates, **kwargs):
                expobj = kwargs['expobj']
                szinvspatial_nontargets = expobj.NonTargetsSzInvasionSpatial

                if type_fr == 'spks - raw':
                    data_mtx = szinvspatial_nontargets.adata.layers['firing_rate_prestim']
                elif type_fr == 'spks - zscored':
                    data_mtx = szinvspatial_nontargets.adata.layers['firing_rate_prestim_zscored']
                elif type_fr == 'gcamp - zscored':
                    data_mtx = szinvspatial_nontargets.adata.layers['raw_gcamp_prestim_zscored']
                elif type_fr == 'neuropil - zscored':
                    data_mtx = szinvspatial_nontargets.adata.layers['neuropil_prestim_zscored']
                else:
                    raise ValueError('type must be `raw` or `zscored` or `gcamp` or `neuropil`')

                # troubleshooting
                pos_dist = np.where(szinvspatial_nontargets.adata.X > 0)
                firing_ratesx = data_mtx[pos_dist]

                # troubleshooting

                for row in range(szinvspatial_nontargets.adata.n_obs):
                    for col in range(szinvspatial_nontargets.adata.n_vars):
                        dist = szinvspatial_nontargets.adata.X[row, col]
                        if not np.isnan(dist):
                            firing_rate = data_mtx[row, col]
                            _bin_nums = [idx for idx, bin in enumerate(rolling_bins[:-20]) if bin[0] < dist < bin[1]]
                            for i in _bin_nums:
                                num[i] += 1
                                y[i] += firing_rate
                                firing_rates[i].append(np.round(firing_rate, 5))

                return num, y, firing_rates

            func_collector = add_dist_responses(rolling_bins=rolling_bins, num=num, y=y, firing_rates=firing_rates)

            num, y, firing_rates = func_collector[-1][0], func_collector[-1][1], func_collector[-1][2]

            avg_firing_rates = np.asarray([np.mean(firing_rates_) for firing_rates_ in firing_rates if len(firing_rates_) > 0])
            non_zeros = np.asarray([np.sum(firing_rates_) for firing_rates_ in firing_rates if len(firing_rates_) > 0])


            # calculate 95% ci for avg responses
            conf_int = np.array([stats.t.interval(alpha=0.95, df=len(responses_) - 1, loc=np.mean(responses_), scale=stats.sem(responses_)) for responses_ in firing_rates if len(responses_) > 0])

            # RUN STATS: 1-WAY ANOVA and KRUSKAL-WALLIS - responses across distance
            kruskal_r = stats.kruskal(*firing_rates[:-1])
            oneway_r = stats.f_oneway(*firing_rates[:-1])

            results.rolling_binned__distance_vs_firingrates[type_fr] = {'bin_width_um': bin_width,
                                                                        'distance_bins': distances,
                                                                        'num_points_in_bin': num,
                                                                        'avg_firing_response_in_bin': avg_firing_rates,
                                                                        'firing rate measure type': type_fr,
                                                                        '95conf_int': conf_int,
                                                                        'all firing rates (per bin)': np.asarray(firing_rates),
                                                                        'kruskal - binned responses': kruskal_r,
                                                                        'anova oneway - binned responses': oneway_r}
            results.save_results()
        kwargs = {}

        # return bin_width, distances, num, avg_responses, conf_int

    # 2.4) PLOT - distance vs. firing rates - no percentile normalization of distances - ROLLING BINS
    @staticmethod
    def plot__responses_v_distance_no_normalization_rolling_bins(results, save_path_full=None, type_fr = "neuropil - zscored", **kwargs):
        """plotting of binned neuropil firing over distance as a step function"""
        # type_fr = "neuropil - zscored"
        data_results = results.rolling_binned__distance_vs_firingrates[type_fr]

        distances = data_results['distance_bins'][:-20]
        avg_firing_rates = data_results['avg_firing_response_in_bin']
        conf_int = data_results['95conf_int']
        num2 = data_results['num_points_in_bin']

        conf_int_distances = pj.flattenOnce([[distances[i], distances[i + 1]] for i in range(len(distances) - 1)])
        conf_int_values_neg = pj.flattenOnce([[val, val] for val in conf_int[1:, 0]])
        conf_int_values_pos = pj.flattenOnce([[val, val] for val in conf_int[1:, 1]])

        #### MAKE PLOT
        fig, ax = (kwargs['fig'], kwargs['axes']) if 'fig' in kwargs or 'axes' in kwargs else plt.subplots(
            figsize=(5, 3), nrows=1, ncols=1, dpi=100)
        # fig, axs = plt.subplots(figsize=(6, 5), nrows=2, ncols=1, dpi=200)

        # ax.plot(distances[:-1], avg_responses, c='cornflowerblue', zorder=1)
        ax.step(distances, avg_firing_rates, c='green', zorder=1, lw=1)
        # ax.fill_between(x=(distances-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
        ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
                        zorder=0)
        ax.step(distances, avg_firing_rates, c='green', zorder=1, lw=1)
        # ax.scatter(distances[:-1], avg_responses, c='orange', zorder=4)
        # ax.set_ylim([-2, 2.25])
        # ax.set_yticks([-1, 0, 1, 2])
        # ax.set_title(
        #     f'photostim responses vs. distance to sz wavefront (binned every {results.rolling_binned__distance_vs_photostimresponses["bin_width_um"]}um)',
        #     wrap=True)
        ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)')
        # ax.set_ylabel(TargetsSzInvasionSpatial_codereview.response_type)
        ax.set_ylabel(type_fr)
        ax.margins(0)
        y_lims = [0, 4.5] if type_fr == 'neuropil - zscored' else ax.get_ylim()
        ax.set_ylim(y_lims)
        if not 'fig' in kwargs and not 'axes' in kwargs:  # fig.tight_layout(pad=1)
            fig.tight_layout()
            fig.show()
            save_path_full = f'{SAVE_FIG}/nontargets-firing_rates-vs-sz_distance_binned_line_plot.png'
            # Utils.save_figure(fig, save_path_full=save_path_full) if save_path_full is not None else None
            # Utils.save_figure(fig, save_path_full=save_path_full[:-4] + '.svg') if save_path_full is not None else None



if __name__ == '__main__':
    results = NonTargetsSzInvasionSpatialResults.load()

    # NonTargetsSzInvasionSpatial.run__classify_nontargets_szboundary()
    # NonTargetsSzInvasionSpatial.run__NonTargetsSzInvasionSpatial()
    # NonTargetsSzInvasionSpatial.run__methods()
    # NonTargetsSzInvasionSpatial.collect__binned__distance_v_firing_rates_rolling_bins(results=results, rerun=1)
    NonTargetsSzInvasionSpatial.plot__responses_v_distance_no_normalization_rolling_bins(results=results)

    """
    TODO:
    - all avg firing rates in all bins are zero. why is this the case? is it because of z scoring? or the raw isn't right?
        dec8 - hunch is that the raw deconvolved spikes are all messed up due to the seizures period. 
        moving on to try move of the raw calcium values next to see what that data might lead something. 
     
    """





