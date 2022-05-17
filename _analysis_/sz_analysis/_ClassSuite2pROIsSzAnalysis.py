import os
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

        self.avg_spk_rate = {}  #: s2p OASIS deconvolved spk rates averaged across experiments
        self.spk_rates = {}  #: s2p OASIS deconvolved spk rates across all cells
        self.neural_activity_rate = {}  #: auc/time of recording of gaussian smoothed and binned s2p OASIS deconvolved spk activity


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
    def collect__avg_spk_rate(Suite2pROIsSzResults: Suite2pROIsSzResults):

        # Gaussian filter smoothing of spks data
        def fwhm2sigma(fwhm):
            return fwhm / np.sqrt(8 * np.log(2))

        def frames2sigma(frames):
            return np.sqrt(frames / 2)

        # binning of spks data
        def rebin(arr, new_shape):
            """Rebin 2D array arr to shape new_shape by averaging."""
            shape = (new_shape[0], arr.shape[0] // new_shape[0],
                     new_shape[1], arr.shape[1] // new_shape[1])
            return arr.reshape(shape).mean(-1).mean(1)

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, set_cache=True, allow_rerun=1)
        def collect__pre4ap_spk_rate(**kwargs):
            expobj: alloptical = kwargs['expobj']
            # expobj = import_expobj(exp_prep='RL108 t-009')

            # calculate spks/s across all cells
            spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'], axis=1) / (
                    expobj.n_frames / expobj.fps)
            avg_spks_per_sec = np.mean(spks_per_sec)

            # Gaussian filter smoothing of spks data
            from scipy.ndimage import gaussian_filter
            spks_smooth_ = np.asarray([gaussian_filter(a, sigma=frames2sigma(frames=int(expobj.fps))) for a in
                                       spks_per_sec])  # TODO this is Matthias's suggested metric for calculating sigma, need to confirm

            # rebinning of spks data
            bin = 4 if int(expobj.fps) == 15 else 8
            spks_smooth_binned = rebin(spks_smooth_, (spks_smooth_.shape[0], int(spks_smooth_.shape[1] / bin)))

            from sklearn.metrics import auc
            area = [auc(np.arange(len(spks_smooth_binned[i])), spks_smooth_binned[i])
                    for i in range(len(spks_smooth_binned))]

            imaging_len_secs = spks_smooth_binned.shape[1] / expobj.fps

            neural_activity_rate = np.array(area) / imaging_len_secs

            # # test plot cumsum plot
            # values, base = np.histogram(spks_per_sec, bins=100)
            #
            # # evaluate the cumulative function
            # cumulative = np.cumsum(values) / len(spks_per_sec)
            #
            # # plot the cumulative function
            # plt.plot(base[:-1], cumulative, c='blue')
            #
            # plt.show()

            return avg_spks_per_sec, spks_per_sec, neural_activity_rate

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=True, allow_rerun=1)
        def collect__interictal_spk_rate(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            # calculate spks/s across all cells
            interictal_idx = [idx for idx, val in enumerate(expobj.Suite2pROIsSz.adata.var['ictal_fr']) if val == False]
            interictal_spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'][:, interictal_idx],
                                             axis=1) / (expobj.n_frames / expobj.fps)
            avg_interictal_spks_per_sec = np.mean(interictal_spks_per_sec)

            # Gaussian filter smoothing of spks data
            from scipy.ndimage import gaussian_filter
            spks_smooth_ = np.asarray([gaussian_filter(a, sigma=frames2sigma(frames=int(expobj.fps))) for a in
                                       interictal_spks_per_sec])  # TODO this is Matthias's suggested metric for calculating sigma, need to confirm

            # rebinning of spks data
            bin = 4 if int(expobj.fps) == 15 else 8
            spks_smooth_binned = rebin(spks_smooth_, (spks_smooth_.shape[0], int(spks_smooth_.shape[1] / bin)))

            from sklearn.metrics import auc
            area = [auc(np.arange(len(spks_smooth_binned[i])), spks_smooth_binned[i])
                    for i in range(len(spks_smooth_binned))]

            imaging_len_secs = spks_smooth_binned.shape[1] / expobj.fps

            neural_activity_rate = np.array(area) / imaging_len_secs

            return avg_interictal_spks_per_sec, interictal_spks_per_sec, neural_activity_rate

        @Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, set_cache=True, allow_rerun=1)
        def collect__ictal_spk_rate(**kwargs):
            expobj: Post4ap = kwargs['expobj']

            # calculate spks/s across all cells
            ictal = [idx for idx, val in enumerate(expobj.Suite2pROIsSz.adata.var['ictal_fr']) if val == True]
            ictal_spks_per_sec = np.sum(expobj.Suite2pROIsSz.adata.layers['s2p_spks'][:, ictal],
                                        axis=1) / (expobj.n_frames / expobj.fps)
            avg_ictal_spks_per_sec = np.mean(ictal_spks_per_sec)

            # Gaussian filter smoothing of spks data
            from scipy.ndimage import gaussian_filter
            spks_smooth_ = np.asarray([gaussian_filter(a, sigma=frames2sigma(frames=int(expobj.fps))) for a in
                                       ictal_spks_per_sec])  # TODO this is Matthias's suggested metric for calculating sigma, need to confirm

            # rebinning of spks data
            bin = 4 if int(expobj.fps) == 15 else 8
            spks_smooth_binned = rebin(spks_smooth_, (spks_smooth_.shape[0], int(spks_smooth_.shape[1] / bin)))

            from sklearn.metrics import auc
            area = [auc(np.arange(len(spks_smooth_binned[i])), spks_smooth_binned[i])
                    for i in range(len(spks_smooth_binned))]

            imaging_len_secs = spks_smooth_binned.shape[1] / expobj.fps

            neural_activity_rate = np.array(area) / imaging_len_secs

            return avg_ictal_spks_per_sec, ictal_spks_per_sec, neural_activity_rate

        pre4ap_spk_rate = np.asarray(collect__pre4ap_spk_rate())[:, 1]
        interictal_spk_rate = np.asarray(collect__interictal_spk_rate())[:, 1]
        ictal_spk_rate = np.asarray(collect__ictal_spk_rate())[:, 1]

        avg_pre4ap_spk_rate = np.asarray(collect__pre4ap_spk_rate())[:, 0]
        avg_interictal_spk_rate = np.asarray(collect__interictal_spk_rate())[:, 0]
        avg_ictal_spk_rate = np.asarray(collect__ictal_spk_rate())[:, 0]

        neural_activity_rate_pre4ap = np.asarray(collect__pre4ap_spk_rate())[:, 2]
        neural_activity_rate_interictal = np.asarray(collect__interictal_spk_rate())[:, 2]
        neural_activity_rate_ictal = np.asarray(collect__ictal_spk_rate())[:, 2]

        Suite2pROIsSzResults.spk_rates = {'baseline': pre4ap_spk_rate,
                                          'interictal': interictal_spk_rate,
                                          'ictal': ictal_spk_rate}

        Suite2pROIsSzResults.avg_spk_rate = {'baseline': avg_pre4ap_spk_rate,
                                             'interictal': avg_interictal_spk_rate,
                                             'ictal': avg_ictal_spk_rate}

        Suite2pROIsSzResults.neural_activity_rate = {'baseline': neural_activity_rate_pre4ap,
                                                     'interictal': neural_activity_rate_interictal,
                                                     'ictal': neural_activity_rate_ictal}

        Suite2pROIsSzResults.save_results()


    @staticmethod
    def plot__avg_spk_rate(pre4ap_spk_rate, interictal_spk_rate):
        # make plot
        pplot.plot_bar_with_points(
            data=[pre4ap_spk_rate, interictal_spk_rate],
            bar=False, x_tick_labels=['baseline', 'interictal'],
            colors=['cornflowerblue', 'forestgreen'], lw=1.3,
            expand_size_x=0.4, title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)', alpha=0.7,
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
    # main.plot__avg_spk_rate(results.avg_spk_rate['pre4ap'], results.avg_spk_rate['interictal'],
    #                         results.avg_spk_rate['ictal'])

# %%

# pre4ap


# post4ap
# expobj = import_expobj(exp_prep='RL108 t-013')
