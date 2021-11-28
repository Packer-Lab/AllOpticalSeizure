# %% IMPORT MODULES AND TRIAL expobj OBJECT
import os; import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-16'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None

# import results superobject
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

# %%
"""######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""
# sys.exit()





# %% 2) plot histogram of zscore stim responses pre and post 4ap and in sz (excluding cells inside sz boundary)

pre_4ap_zscores = []
post_4ap_zscores = []
in_sz_zscores = []
for prep in allopticalResults.stim_responses_zscores.keys():
    count = 0
    for i in allopticalResults.pre_4ap_trials:
        if prep in i[0]:
            count += 1

    for key in list(allopticalResults.stim_responses_zscores[prep].keys()):
        # comparison_number += 1
        # key = ls(allopticalResults.stim_responses_zscores[prep].keys())[comparison_number]
        trial_comparison = allopticalResults.stim_responses_zscores[prep][key]
        if 'pre-4ap' in trial_comparison.keys():
            pre_4ap_response_df = trial_comparison['pre-4ap']
            for col in pre_4ap_response_df.columns:
                if 'z' in str(col):
                    pre_4ap_zscores = pre_4ap_zscores + list(pre_4ap_response_df[col][:-2])

        if 'post-4ap' in trial_comparison.keys():
            post_4ap_response_df = trial_comparison['post-4ap']
            for col in post_4ap_response_df.columns:
                if 'z' in str(col):
                    post_4ap_zscores = post_4ap_zscores + list(post_4ap_response_df[col][:-2])

        if 'in sz' in trial_comparison.keys():
            in_sz_response_df = trial_comparison['in sz']
            for col in in_sz_response_df.columns:
                if 'z' in str(col):
                    in_sz_zscores = in_sz_zscores + list(in_sz_response_df[col][:-2])

in_sz_zscores = [score for score in in_sz_zscores if str(score) != 'nan']
data = [pre_4ap_zscores, in_sz_zscores, post_4ap_zscores]
pj.plot_hist_density(data, x_label='z-score', title='All exps. stim responses zscores (normalized to pre-4ap)',
                     fill_color=['green', '#ff9d09', 'steelblue'], num_bins=1000, show_legend=True, alpha=1.0, mean_line=True,
                     figsize=(4, 4), legend_labels=['pre 4ap', 'ictal', 'interictal'], x_lim=[-15, 15])



# %% 8.0-dc) DATA COLLECTION - zscore of stim responses vs. TIME to seizure onset - for loop over all experiments to collect zscores in terms of sz onset time

stim_relative_szonset_vs_avg_zscore_alltargets_atstim = {}

for prep in allopticalResults.stim_responses_zscores.keys():
    # prep = 'PS07'

    for key in list(allopticalResults.stim_responses_zscores[prep].keys()):
        # key = list(allopticalResults.stim_responses_zscores[prep].keys())[0]
        # comp = 2
        if 'post-4ap' in allopticalResults.stim_responses_zscores[prep][key]:
            post_4ap_response_df = allopticalResults.stim_responses_zscores[prep][key]['post-4ap']
            if len(post_4ap_response_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, post4ap trial: {post4aptrial}')
                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"] = [[], []]
                expobj, experiment = aoutils.import_expobj(trial=post4aptrial, prep=prep, verbose=False)

                # transform the rows of the stims responses dataframe to relative time to seizure
                stims = list(post_4ap_response_df.index)
                stims_relative_sz = []
                for stim_idx in stims:
                    stim_frame = expobj.stim_start_frames[stim_idx]
                    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
                    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
                    stims_relative_sz.append(round(time_diff, 3))

                cols = [col for col in post_4ap_response_df.columns if 'z' in str(col)]
                post_4ap_response_df_zscore_stim_relative_to_sz = post_4ap_response_df[cols]
                post_4ap_response_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored response_df and assign a new index where the col names are times relative to sz onset

                # take average of all targets at a specific time to seizure onset
                post_4ap_response_df_zscore_stim_relative_to_sz['avg'] = post_4ap_response_df_zscore_stim_relative_to_sz.T.mean()

                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(stims_relative_sz)
                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(post_4ap_response_df_zscore_stim_relative_to_sz['avg'].tolist())


allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim = stim_relative_szonset_vs_avg_zscore_alltargets_atstim
allopticalResults.save()
