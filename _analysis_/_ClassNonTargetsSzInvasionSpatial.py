import sys

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

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"

# %% ###### NON TARGETS analysis + plottings

class NonTargetsSzInvasionSpatialResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__PhotostimResponsesNonTargets.pkl'

    def __init__(self):
        super().__init__()

        self.summed_responses = None  #: dictionary of baseline and interictal summed responses of targets and nontargets across experiments
        self.lin_reg_summed_responses = None  #: dictionary of baseline and interictal linear regression metrics for relating total targets responses and nontargets responses across experiments
        self.avg_responders_num = None  #: average num responders, for pairedmatched experiments between baseline pre4ap and interictal
        self.avg_responders_magnitude = None  #: average response magnitude, for pairedmatched experiments between baseline pre4ap and interictal

REMAKE = False
if not os.path.exists(NonTargetsSzInvasionSpatialResults.SAVE_PATH) or REMAKE:
    results = NonTargetsSzInvasionSpatialResults()
    results.save_results()



class NonTargetsSzInvasionSpatial(Quantification):
    """class for classifying nontargets relative to sz boundary and quantifying distance to sz boundary.

    Tasks:
    [x] code for splitting nontargets inside and outside sz boundary - during ictal stims
    [ ] run through for all experiments - transfer over from the NontargetsPhotostimResponsesQuant class

    """

    save_path = SAVE_LOC + 'NonTargetsSzInvasionSpatial.pkl'
    EXCLUDE_TRIALS = ['PS04 t-012',  # no responding cells and also doubled up by PS04 t-017 in any case
                      ]

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)

        distance_to_sz = self.classify_and_measure_nontargets_szboundary(expobj=expobj, force_redo=True)
        self.adata: AnnotatedData2 = self.create_anndata(expobj=expobj, distance_to_sz=distance_to_sz) #: anndata table to hold data about distances to seizure boundary for nontargets

        print(f'\- ADDING NEW NonTargetsSzInvasionSpatial MODULE to expobj: {expobj.t_series_name}')


    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1, skip_trials=EXCLUDE_TRIALS)
    def run__initNonTargetsSzInvasionSpatial(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.NonTargetsSzInvasionSpatial = NonTargetsSzInvasionSpatial(expobj=expobj)
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
    @Utils.run_for_loop_across_exps(run_post4ap_trials=True)
    def run__classify_nontargets_szboundary(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        expobj.NonTargetsSzInvasionSpatial._classify_nontargets_szboundary(expobj=expobj, force_redo=True)
        expobj.save()


    def _calc_min_distance_sz_nontargets(self, expobj: Post4ap, force_redo=False):
        assert hasattr(expobj.ExpSeizure, 'nontargets_szboundary_stim')

        if not force_redo:
            return

        distance_to_sz_df = expobj.calcMinDistanceToSz_newer(analyse_cells='s2p nontargets', show_debug_plot=False)

        assert distance_to_sz_df.shape == self.adata.shape

        return distance_to_sz_df / expobj.pix_sz_x


    def classify_and_measure_nontargets_szboundary(self, force_redo=False, **kwargs):
        expobj: Post4ap = kwargs['expobj']
        self._classify_nontargets_szboundary(expobj=expobj, force_redo=force_redo)
        expobj.save()
        distance_to_sz_um = expobj.PhotostimResponsesNonTargets._calc_min_distance_sz_nontargets(expobj=expobj, force_redo=force_redo)
        return distance_to_sz_um


    # 1) CREATE ANNDATA - using distance to sz dataframe collected above ###############################################
    def create_anndata(self, expobj: Union[alloptical, Post4ap], distance_to_sz):
        """
        Creates annotated data (see anndata library for more information on AnnotatedData) object based around the photostim resposnes of all non-target ROIs.

        """

        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from non targets meta information

        obs_meta = pd.DataFrame(
            columns=['original_index', 'footprint', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'radius',
                     'aspect_ratio', 'npix_norm', 'skew', 'std'], index=range((len(expobj.s2p_nontargets)) - len(expobj.s2p_nontargets_exclude)))
        for i, idx in enumerate(obs_meta.index):
            if idx not in expobj.s2p_nontargets_exclude:
                for __column in obs_meta:
                    obs_meta.loc[i, __column] = expobj.stat[i][__column]


        # build numpy array for multidimensional obs metadata
        obs_m = {'ypix': [],
                 'xpix': []}
        for col in [*obs_m]:
            for i, idx in enumerate(expobj.s2p_nontargets):
                if idx not in expobj.s2p_nontargets_exclude:
                    obs_m[col].append(expobj.stat[i][col])
            obs_m[col] = np.asarray(obs_m[col])


        var_meta = pd.DataFrame(index=['stim_group', 'im_time_secs', 'stim_start_frame', 'wvfront in sz', 'seizure location'],
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
        photostim_responses_adata = AnnotatedData2(X=distance_to_sz, obs=obs_meta, var=var_meta.T, obsm=obs_m,
                                                   data_label='distance to sz (um)')

        print(f"Created: {photostim_responses_adata}")
        return photostim_responses_adata



    def _add_nontargets_sz_boundary_anndata(self):
        """add layer to anndata table that splits nontarget cell in or out of sz boundary.
        1: outside of sz boundary
        0: inside of sz boundary

        """

        arr = np.empty_like(self.adata.X)

        arr[np.where(self.adata.X > 0)] = 1
        arr[np.where(self.adata.X < 0)] = 0

        self.adata.add_layer(layer_name='in/out sz', data=arr)

if __name__ == '__main__':
    NonTargetsSzInvasionSpatial.run__initNonTargetsSzInvasionSpatial()

    # NonTargetsSzInvasionSpatial.run__classify_nontargets_szboundary()

