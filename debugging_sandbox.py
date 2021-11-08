# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')


# %run ./setup_notebook.ipynb
# print(sys.path)

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
# sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/'


########

# # %% 5.1) for loop to go through each expobj to analyze nontargets - post4ap trials
ls = ['RL108 t-013', 'RL109 t-021', 'RL109 t-016']
# ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
for key in list(allopticalResults.trial_maps['post'].keys()):
    for j in range(len(allopticalResults.trial_maps['post'][key])):

        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))
        if expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] in ls:
            aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', do_processing=True, to_plot=True,
                                                     save_plot_suffix=f"Nontargets_responses_2021-11-08/{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-post4ap.png")
        else:
            pass
            # aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', do_processing=False, to_plot=False,
            #                                          save_plot_suffix=f"Nontargets_responses_2021-11-06/{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-post4ap.png")






# %%
# 5.2.1.1) scatter plot response magnitude vs. prestim std F
ls = ['post']
for i in ls:
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
    counter = 0

    j = 0
    for key in list(allopticalResults.trial_maps[i].keys()):

        expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')  # import expobj

        possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
        negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
        nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)

        stims = [i for i, stim in enumerate(expobj.stim_start_frames) if stim not in expobj.stims_in_sz]
        stims_sz = [i for i, stim in enumerate(expobj.stim_start_frames) if stim in list(expobj.slmtargets_sz_stim.keys())]

        posunits_prestdF = np.mean(np.std(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[:, stims][expobj.sig_units], axis=1) > 0)[0], :, :][:, stims, :], axis=2), axis=1)
        negunits_prestdF = np.mean(np.std(expobj.raw_traces[np.where(np.nanmean(expobj.post_array_responses[:, stims][expobj.sig_units], axis=1) < 0)[0], :, :][:, stims, :], axis=2), axis=1)
        nonsigunits_prestdF = np.mean(np.std(expobj.raw_traces[~expobj.sig_units, stims, :], axis=2), axis=1)

        assert len(possig_responders_avgresponse) == len(posunits_prestdF)
        assert len(negsig_responders_avgresponse) == len(negunits_prestdF)
        assert len(nonsig_responders_avgresponse) == len(nonsigunits_prestdF)

        # fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10))
        ax = axs[counter // ncols, counter % ncols]
        ax.scatter(x = nonsigunits_prestdF, y = nonsig_responders_avgresponse, color='gray', alpha=0.10, label='non sig.', s=65, edgecolors='none', zorder=0)
        ax.scatter(x = negunits_prestdF, y = negsig_responders_avgresponse, color='red', alpha=0.10, label='sig. neg.', s=65, edgecolors='none', zorder=1)
        ax.scatter(x = negunits_prestdF, y = possig_responders_avgresponse, color='green', alpha=0.10, label='sig. pos.', s=65, edgecolors='none', zorder=2)
        ax.set_title(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} ")
        # fig.show()

        counter += 1
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Avg. prestim std F')
    axs[0, 0].set_ylabel('Avg. mag (dF/stdF)')
    fig.tight_layout()
    fig.suptitle(f'All exps. prestim std F vs. response mag (dF/stdF) distribution - {i}4ap', y = 0.98)
    save_path = expobj.analysis_save_path[:30] + 'Results_figs/' + \
                f"Nontargets_responses_2021-11-06/scatter prestim std F vs. plot response magnitude - {i}4ap.png"
    plt.savefig(save_path)
    fig.show()