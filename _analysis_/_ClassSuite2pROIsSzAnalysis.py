import os
from typing import Union

import numpy as np
import pandas as pd

import _alloptical_utils as Utils
from funcsforprajay import plotting as pplot

from _analysis_._utils import Quantification, Results
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_._anndata import AnnotatedData2
from _utils_.io import import_expobj

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# %% results

class Suite2pROIsSzResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__Suite2pROIsSz.pkl'

    def __init__(self):
        super().__init__()

        self.avg_spk_rate = {}

REMAKE = False
if not os.path.exists(Suite2pROIsSzResults.SAVE_PATH) or REMAKE:
    results = Suite2pROIsSzResults()
    results.save_results()




class Suite2pROIsSz(Quantification):
    """class for analysis of suite2p processed data in relation to seizures (still including baseline conditions though)

    [x]: bar graph of baseline, interictal, ictal suite2p ROIs activity
    todo: bar graph of ROIs' avg corr. between neuropil and firing activity, across baseline, interictal, and ictal

    """
    SAVE_PATH = SAVE_LOC + 'Suite2pROIsSz.pkl'

    normalization_type = 'dFF (zscored) (interictal)'

    def __init__(self, expobj: Union[alloptical, Post4ap]):
        super().__init__(expobj)
        print(f'\- ADDING NEW Suite2pROIsSz MODULE to expobj: {expobj.t_series_name}')
        self.s2pPath = expobj.s2p_path
        self.adata: AnnotatedData2 = self._create_anndata(expobj=expobj)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=0)
    def run__init_Suite2pROIsSz(**kwargs):
        """loop for constructor for all experiments"""
        expobj = kwargs['expobj']
        expobj.Suite2pROIsSz = Suite2pROIsSz(expobj=expobj)
        expobj.save()

    def __repr__(self):
        return f"Suite2pROIsSz <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"

    def loadSuite2pResults(self, expobj: Union[alloptical, Post4ap]):
        s2pFminusFneu = expobj.raw
        s2pSpks = expobj.spks
        s2pNeuropil = expobj.neuropil
        s2pStat = expobj.stat

        return s2pFminusFneu, s2pSpks, s2pNeuropil, s2pStat

    # 0) CREATE ANNDATA OBJECT TO STORE SUITE2P DATA
    def _create_anndata(self, expobj: Post4ap):
        """
        Creates annotated data by extending the adata table from .PhotostimResponsesSLMTargets.adata and adding the
        time to sz onset for each stim frame as a variable.

        """
        s2pFminusFneu, s2pSpks, s2pNeuropil, s2pStat = self.loadSuite2pResults(expobj=expobj)


        # SETUP THE OBSERVATIONS (CELLS) ANNOTATIONS TO USE IN anndata
        # build dataframe for obs_meta from suite2p stat information
        obs_meta = pd.DataFrame(
            columns=['original_index', 'footprint', 'mrs', 'mrs0', 'compact', 'med', 'npix', 'radius',
                     'aspect_ratio', 'npix_norm', 'skew', 'std'], index=range(len(s2pStat)))
        for idx, __stat in enumerate(s2pStat):
            for __column in obs_meta:
                obs_meta.loc[idx, __column] = __stat[__column]

        # build numpy array for multidimensional obs metadata
        obs_m = {'ypix': [],
                 'xpix': []}
        for col in [*obs_m]:
            for idx, __stat in enumerate(s2pStat):
                obs_m[col].append(__stat[col])
            obs_m[col] = np.asarray(obs_m[col])

        # SETUP THE VARIABLES ANNOTATIONS TO USE IN anndata
        # build dataframe for var annot's from Paq file
        var_meta = pd.DataFrame(index=['photostim_frame'], columns=range(expobj.n_frames))
        stim_frames_ = [False for i in range(expobj.n_frames)]
        for frame in expobj.photostim_frames:
            stim_frames_[frame] = True
        var_meta.loc['photostim_frame'] = stim_frames_

        # BUILD LAYERS TO ADD TO anndata OBJECT
        if not hasattr(expobj, 'dFF'):
            dff = expobj.dfof()
        else:
            dff = expobj.dFF

        layers = {'raw_dFF_normalized': dff,
                  's2p_spks': s2pSpks,
                  's2p_neuropil': s2pNeuropil
                  }

        print(f"\n\----- CREATING annotated data object using AnnData:")
        _data_type = 'Suite2p Raw (neuropil substracted)'
        adata = AnnotatedData2(X=s2pFminusFneu, obs=obs_meta, var=var_meta.T, obsm=obs_m, layers=layers,
                                 data_label=_data_type)

        print(f"\n{adata}")
        return adata

    # 0.1) ADD INTERICTAL VS. ICTAL IMAGING FRAMES LABEL TO ANNDATA
    @staticmethod
    @Utils.run_for_loop_across_exps(run_post4ap_trials=True, allow_rerun=0)
    def label__interictal_ictal_post4ap(**kwargs):
        expobj: Post4ap = kwargs['expobj']
        ictal_fr = [False for i in range(expobj.n_frames)]
        for frame in expobj.seizure_frames:
            ictal_fr[frame] = True

        expobj.Suite2pROIsSz.adata.add_variable(var_name='ictal_fr', values=ictal_fr)
        expobj.save()

    # 1) MEASURING + PLOTTING AVG SPIKES RATE FOR EXPERIMENTS
    @staticmethod
    def collect__avg_spk_rate():
        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=True)
        def collect__pre4ap_spk_rate(**kwargs):
            expobj = kwargs['expobj']
            # expobj = import_expobj(exp_prep='RL108 t-009')

            # calculate spks/s across all cells
            spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'], axis=1) / (
                        expobj.n_frames / expobj.fps)
            avg_spks_per_sec = np.mean(spks_per_sec)
            return avg_spks_per_sec

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=True)
        def collect__post4ap_spk_rate(**kwargs):
            expobj = kwargs['expobj']

            # calculate spks/s across all cells
            interictal_idx = [idx for idx, val in enumerate(expobj.Suite2pROIsSz.adata.var['ictal_fr']) if val == False]
            interictal_spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'][:, interictal_idx],
                                             axis=1) / (expobj.n_frames / expobj.fps)
            avg_interictal_spks_per_sec = np.mean(interictal_spks_per_sec)

            # calculate spks/s across all cells
            ictal = [idx for idx, val in enumerate(expobj.Suite2pROIsSz.adata.var['ictal_fr']) if val == True]
            ictal_spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'][:, ictal],
                                             axis=1) / (expobj.n_frames / expobj.fps)
            avg_ictal_spks_per_sec = np.mean(ictal_spks_per_sec)

            return avg_interictal_spks_per_sec, avg_ictal_spks_per_sec

        pre4ap_spk_rate = collect__pre4ap_spk_rate()
        func_collector = collect__post4ap_spk_rate()

        interictal_spk_rate, ictal_spk_rate = np.asarray(func_collector)[:, 0], np.asarray(func_collector)[:, 1]

        return pre4ap_spk_rate, interictal_spk_rate, ictal_spk_rate

    @staticmethod
    def plot__avg_spk_rate(pre4ap_spk_rate, interictal_spk_rate, ictal_spk_rate):
        # make plot
        pplot.plot_bar_with_points(
            data=[pre4ap_spk_rate, interictal_spk_rate, ictal_spk_rate],
            bar=True, x_tick_labels=['baseline', 'interictal', 'ictal'],
            colors=['blue', 'green', 'purple'], lw=1.3,
            expand_size_x=0.4, title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)',
            expand_size_y=1.2)




# %% run analysis/results code

if __name__ == '__main__':
    main = Suite2pROIsSz
    results: Suite2pROIsSzResults = Suite2pROIsSzResults.load()

    main.run__init_Suite2pROIsSz()
    main.label__interictal_ictal_post4ap()

    # pre4ap_spk_rate, interictal_spk_rate, ictal_spk_rate = main.collect__avg_spk_rate()

    # results.avg_spk_rate = {'pre4ap': pre4ap_spk_rate,
    #                         'interictal': interictal_spk_rate,
    #                         'ictal': ictal_spk_rate}
    # results.save_results()
    main.plot__avg_spk_rate(results.avg_spk_rate['pre4ap'], results.avg_spk_rate['interictal'],
                            results.avg_spk_rate['ictal'])


# %%

# pre4ap




# post4ap
# expobj = import_expobj(exp_prep='RL108 t-013')



