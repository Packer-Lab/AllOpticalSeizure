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


sys.exit()
"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


# %% 1) DATA COLLECTION: organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
allopticalResults.stim_responses_zscores = {}
for i in range(len(allopticalResults.pre_4ap_trials)):
    prep = allopticalResults.pre_4ap_trials[i][0][:-6]
    pre4aptrial = allopticalResults.pre_4ap_trials[i][0][-5:]
    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                prep, pre4aptrial), 'date'])[0]
    print(f"\n{i}, {date}, {prep}")


    # skipping some trials that need fixing of the expobj
    if f"{prep} {pre4aptrial}" not in trials_skip:


        # load up pre-4ap trial
        print(f'|-- importing {prep} {pre4aptrial} - pre4ap trial')



        expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep, verbose=False)

        df = expobj.responses_SLMtargets.T  # df == stim frame x cells (photostim targets)
        if len(allopticalResults.pre_4ap_trials[i]) > 1:
            for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
                print(f"|-- {i}, {j}")
                # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                prep = allopticalResults.pre_4ap_trials[i][j][:-6]
                pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
                if f"{prep} {pre4aptrial}" not in trials_skip:
                    print(f"adding trial to this comparison: {pre4aptrial_} [1.0]")
                    date = list(allopticalResults.slmtargets_stim_responses.loc[
                                    allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'|-- importing {prep} {pre4aptrial_} - pre4ap trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False)
                    df_ = expobj.responses_SLMtargets.T

                    # append additional dataframe to the first dataframe
                    df.append(df_, ignore_index=True)
                else:
                    print(f"\-- ***** skipping: {prep} {pre4aptrial_}")

        cols = list(df.columns)  # cols = cells
        # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
        for col in cols:
            col_zscore = str(col) + '_z'
            df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
        # -- add a mean and std calculation for each cell to use for the post-4ap trial scores
        mean = pd.Series(df.mean(), name='mean')
        std = pd.Series(df.std(ddof=0), name='std')
        df = df.append([mean, std])

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in allopticalResults.stim_responses_zscores.keys():
            allopticalResults.stim_responses_zscores[prep] = {}
            comparison_number = 1
        else:
                comparison_number = len(allopticalResults.stim_responses_zscores[prep]) + 1

        allopticalResults.stim_responses_zscores[prep][f'{comparison_number}'] = {'pre-4ap': {}}
        allopticalResults.stim_responses_zscores[prep][f'{comparison_number}']['pre-4ap'] = df

        # allopticalResults.save()


        # expobj.responses_SLMtargets_zscore = df
        # expobj.save()

        pre_4ap_df = df


    else:
        print(f"|-- skipping: {prep} {pre4aptrial}")


    ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS - zscore to the mean and std of the same SLM target calculated from the pre-4ap trial
    post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:]



    # skipping some trials that need fixing of the expobj
    if f"{prep} {post4aptrial}" not in trials_skip:
        print(f'TEST 1.1 - working on {prep} {post4aptrial}')



        # load up post-4ap trial and stim responses
        print(f'|-- importing {prep} {post4aptrial} - post4ap trial')
        expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False)
        if hasattr(expobj, 'responses_SLMtargets_outsz'):
            df = expobj.responses_SLMtargets_outsz.T

            if len(allopticalResults.post_4ap_trials[i]) > 1:
                for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
                    print(f"|-- {i}, {j}")
                    # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                    prep = allopticalResults.post_4ap_trials[i][j][:-6]
                    post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                    if f"{prep} {post4aptrial_}" not in trials_skip:
                        print(f"adding trial to this comparison: {post4aptrial} [1.1]")
                        date = list(allopticalResults.slmtargets_stim_responses.loc[
                                        allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'|-- importing {prep} {post4aptrial_} - post4ap trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                        if hasattr(expobj, 'responses_SLMtargets_outsz'):
                            df_ = expobj.responses_SLMtargets_outsz.T
                            # append additional dataframe to the first dataframe
                            df.append(df_, ignore_index=True)
                        else:
                            print('|-- **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\-- ***** skipping: {prep} {post4aptrial_}")

            cols = list(df.columns)
            # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
            # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP DF
            for col in cols:
                col_zscore = str(col) + '_z'
                df[col_zscore] = (df[col] - pre_4ap_df.loc['mean', col])/pre_4ap_df.loc['std', col]

            allopticalResults.stim_responses_zscores[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** 1 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))



    ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells - zscore to the mean and std of the same SLM target calculated from the pre-4ap trial
    # post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:] -- same as run_post4ap_trials OUTSZ for loop one above



    # skipping some trials that need fixing of the expobj
    # if f"{prep} {post4aptrial}" not in trials_skip:
    #     print(f'TEST 1.2 - working on {prep} {post4aptrial}')

    # using the same skip statement as in the main for loop here

        # load up post-4ap trial and stim responses
        # expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False)  --- dont need to load up
        if hasattr(expobj, 'slmtargets_szboundary_stim'):
            if hasattr(expobj, 'responses_SLMtargets_insz'):
                df = expobj.responses_SLMtargets_insz.T


                # switch to NA for stims for cells which are classified in the sz
                # collect stim responses with stims excluded as necessary
                for target in df.columns:
                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                        if target in expobj.slmtargets_szboundary_stim[stim]:
                            df.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                    # responses = [expobj.responses_SLMtargets_insz.loc[col][expobj.stim_start_frames.index(stim)] for stim in expobj.stims_in_sz if
                    #              col not in expobj.slmtargets_szboundary_stim[stim]]
                    # targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)


                if len(allopticalResults.post_4ap_trials[i]) > 1:
                    for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
                        print(f"|-- {i}, {j}")
                        # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                        prep = allopticalResults.post_4ap_trials[i][j][:-6]
                        post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                        if f"{prep} {post4aptrial_}" not in trials_skip:
                            print(f"{post4aptrial} [1.2]")
                            date = list(allopticalResults.slmtargets_stim_responses.loc[
                                            allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
                                                prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                            if hasattr(expobj, 'responses_SLMtargets_insz'):
                                df_ = expobj.responses_SLMtargets_insz.T

                                # append additional dataframe to the first dataframe
                                # switch to NA for stims for cells which are classified in the sz
                                # collect stim responses with stims excluded as necessary
                                for target in df.columns:
                                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                                        if target in expobj.slmtargets_szboundary_stim[stim]:
                                            df_.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                                df.append(df_, ignore_index=True)
                            else:
                                print(
                                    '**** 4 need to run collecting in sz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} {post4aptrial_}")

                cols = list(df.columns)
                # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
                # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP DF
                for col in cols:
                    col_zscore = str(col) + '_z'
                    df[col_zscore] = (df[col] - pre_4ap_df.loc['mean', col]) / pre_4ap_df.loc['std', col]

                allopticalResults.stim_responses_zscores[prep][f"{comparison_number}"]['in sz'] = df
            else:
                print('**** 4 need to run collecting insz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
                allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
        else:
            print(f"**** 5 need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial}")

    else:
        print(f"\-- ***** skipping: {prep} {post4aptrial}")
        if not hasattr(expobj, 'responses_SLMtargets_outsz'):
            print(f'\-- **** 1 need to run collecting outsz responses SLMtargets attr for {post4aptrial}, {prep} ****')

        if not hasattr(expobj, 'slmtargets_szboundary_stim'):
            print(f'**** 2 need to run collecting insz responses SLMtargets attr for {post4aptrial}, {prep} ****')
        if hasattr(expobj, 'responses_SLMtargets_insz'):
            print(f'**** 3 need to run collecting in sz responses SLMtargets attr for {post4aptrial}, {prep} ****')

    ## switch out this comparison_number to something more readable
    new_key = f"{pre4aptrial} vs. {post4aptrial}"
    allopticalResults.stim_responses_zscores[prep][new_key]= allopticalResults.stim_responses_zscores[prep].pop(f'{comparison_number}')
    # allopticalResults.stim_responses_zscores[prep][new_key]= allopticalResults.stim_responses_zscores[prep][f'{comparison_number}']


allopticalResults.save()



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
            pre_4ap_df = trial_comparison['pre-4ap']
            for col in pre_4ap_df.columns:
                if 'z' in str(col):
                    pre_4ap_zscores = pre_4ap_zscores + list(pre_4ap_df[col][:-2])

        if 'post-4ap' in trial_comparison.keys():
            post_4ap_df = trial_comparison['post-4ap']
            for col in post_4ap_df.columns:
                if 'z' in str(col):
                    post_4ap_zscores = post_4ap_zscores + list(post_4ap_df[col][:-2])

        if 'in sz' in trial_comparison.keys():
            in_sz_df = trial_comparison['in sz']
            for col in in_sz_df.columns:
                if 'z' in str(col):
                    in_sz_zscores = in_sz_zscores + list(in_sz_df[col][:-2])

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
            post_4ap_df = allopticalResults.stim_responses_zscores[prep][key]['post-4ap']
            if len(post_4ap_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, post4ap trial: {post4aptrial}')
                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"] = [[], []]
                expobj, experiment = aoutils.import_expobj(trial=post4aptrial, prep=prep, verbose=False)

                # transform the rows of the stims responses dataframe to relative time to seizure
                stims = list(post_4ap_df.index)
                stims_relative_sz = []
                for stim_idx in stims:
                    stim_frame = expobj.stim_start_frames[stim_idx]
                    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
                    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
                    stims_relative_sz.append(round(time_diff, 3))

                cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
                post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
                post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

                # take average of all targets at a specific time to seizure onset
                post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(stims_relative_sz)
                stim_relative_szonset_vs_avg_zscore_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(post_4ap_df_zscore_stim_relative_to_sz['avg'].tolist())


allopticalResults.stim_relative_szonset_vs_avg_zscore_alltargets_atstim = stim_relative_szonset_vs_avg_zscore_alltargets_atstim
allopticalResults.save()







# %% aoresults-7-dc) zscore of stim responses vs. TIME to seizure onset - original code for single experiments
prep = 'RL108'
date = '2020-12-18'
trial = 't-013'
expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)
post_4ap_df = expobj.responses_SLMtargets_zscore

# transform the rows of the stims responses dataframe to relative time to seizure

stims = list(post_4ap_df.index)
stims_relative_sz = []
for stim_idx in stims:
    stim_frame = expobj.stim_start_frames[stim_idx]
    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
    stims_relative_sz.append(round(time_diff, 3))

cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x=post_4ap_df_zscore_stim_relative_to_sz.index, y=post_4ap_df_zscore_stim_relative_to_sz['avg'])
fig.show()