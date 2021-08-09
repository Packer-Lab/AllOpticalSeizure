# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import results superobject

results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %% convert stim responses to z-scores - relative to pre-4ap scores
# make sure to compare the appropriate pre and post 4ap trial comparisons

allopticalResults.outsz_missing = []
# allopticalResults.stim_responses_zscores = {}
for i in range(len(allopticalResults.pre_4ap_trials)):
    prep = allopticalResults.pre_4ap_trials[i][0][:-6]
    pre4aptrial = allopticalResults.pre_4ap_trials[i][0][-5:]
    date = list(allopticalResults.slmtargets_stim_responses.loc[
                allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
                prep, pre4aptrial), 'date'])[0]
    print(i, date, prep)

    # load up pre-4ap trial
    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep)

    df = expobj.responses_SLMtargets.T  # df == stim frame x cells (photostim targets)
    if len(allopticalResults.pre_4ap_trials[i]) > 1:
        for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
            print(i, j)
            # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
            prep = allopticalResults.pre_4ap_trials[i][j][:-6]
            pre4aptrial = allopticalResults.pre_4ap_trials[i][j][-5:]
            print(pre4aptrial)
            date = list(allopticalResults.slmtargets_stim_responses.loc[
                            allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
                                prep, pre4aptrial), 'date'])[0]

            # load up pre-4ap trial
            expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep)
            df_ = expobj.responses_SLMtargets.T

            # append additional dataframe to the first dataframe
            df.append(df_, ignore_index=True)



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

    allopticalResults.stim_responses_zscores[prep]['%s' % comparison_number] = {'pre-4ap': {}, 'post-4ap': {}}
    allopticalResults.stim_responses_zscores[prep]['%s' % comparison_number]['pre-4ap'] = df

    allopticalResults.save()


    # expobj.responses_SLMtargets_zscore = df
    # expobj.save()

    pre_4ap_df = df


    ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS - zscore to the mean and std of the same SLM target calculated from the pre-4ap trial
    post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:]
    # load up post-4ap trial and stim responses
    expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep)
    if hasattr(expobj, 'outsz_responses_SLMtargets'):
        df = expobj.outsz_responses_SLMtargets.T
    else:
        print('**** need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
        allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))

    if len(allopticalResults.post_4ap_trials[i]) > 1:
        for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
            print(i, j)
            # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
            prep = allopticalResults.post_4ap_trials[i][j][:-6]
            post4aptrial = allopticalResults.post_4ap_trials[i][j][-5:]
            print(post4aptrial)
            date = list(allopticalResults.slmtargets_stim_responses.loc[
                            allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
                                prep, pre4aptrial), 'date'])[0]

            # load up post-4ap trial and stim responses
            expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep)
            if hasattr(expobj, 'outsz_responses_SLMtargets'):
                df_ = expobj.outsz_responses_SLMtargets.T
            else:
                print('**** need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
                allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))

            # append additional dataframe to the first dataframe
            df.append(df_, ignore_index=True)


    cols = list(df.columns)
    for col in cols:
        col_zscore = str(col) + '_z'
        df[col_zscore] = (df[col] - pre_4ap_df.loc['mean', col])/pre_4ap_df.loc['std', col]

    allopticalResults.stim_responses_zscores[prep]['%s' % comparison_number]['post-4ap'] = df




    ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells - zscore to the mean and std of the same SLM target calculated from the pre-4ap trial
    post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:]
    # load up post-4ap trial and stim responses
    expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep)
    if hasattr(expobj, 'insz_responses_SLMtargets'):
        df = expobj.insz_responses_SLMtargets.T
    else:
        print('**** need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
        allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))

    if len(allopticalResults.post_4ap_trials[i]) > 1:
        for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
            print(i, j)
            # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
            prep = allopticalResults.post_4ap_trials[i][j][:-6]
            post4aptrial = allopticalResults.post_4ap_trials[i][j][-5:]
            print(post4aptrial)
            date = list(allopticalResults.slmtargets_stim_responses.loc[
                            allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
                                prep, pre4aptrial), 'date'])[0]

            # load up post-4ap trial and stim responses
            expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep)
            if hasattr(expobj, 'insz_responses_SLMtargets'):
                df_ = expobj.insz_responses_SLMtargets.T
            else:
                print(
                    '**** need to run collecting insz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
                allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))

            # append additional dataframe to the first dataframe
            df.append(df_, ignore_index=True)

    cols = list(df.columns)
    for col in cols:
        col_zscore = str(col) + '_z'
        df[col_zscore] = (df[col] - pre_4ap_df.loc['mean', col]) / pre_4ap_df.loc['std', col]

    allopticalResults.stim_responses_zscores[prep]['%s' % comparison_number]['post-4ap'] = df


    # expobj.responses_SLMtargets_zscore = df
    # expobj.save()

    # post_4ap_df = df

# %% plot histogram of zscore stim responses pre and post 4ap
# prep = 'RL109'
# date = '2020-12-19'


# # post 4ap df
# posttrial = post4aptrial
# expobj, experiment = aoutils.import_expobj(trial=posttrial, date=date, prep=prep)
# post_4ap_df = expobj.responses_SLMtargets_zscore
#
# # pre 4ap df
# pretrial = pre4aptrial
# expobj, experiment = aoutils.import_expobj(trial=pretrial, date=date, prep=prep)
# pre_4ap_df = expobj.responses_SLMtargets_zscore

# pre_4ap_zscores = []
# for col in pre_4ap_df.columns:
#     if 'z' in str(col):
#         pre_4ap_zscores = pre_4ap_zscores + list(pre_4ap_df[col][:-2])
#
# post_4ap_zscores = []
# for col in post_4ap_df.columns:
#     if 'z' in str(col):
#         post_4ap_zscores = post_4ap_zscores + list(post_4ap_df[col][:-2])
#
# data = [pre_4ap_zscores, post_4ap_zscores]
# pj.plot_hist_density(data, x_label='z-score', title='%s stim zscores (normalized to pre-4ap) - pre vs. post' % prep, fill_color=['blue', 'orange'],
#                      figsize=(5,4), legend_labels=['pre-4ap %s' % pretrial, 'post-4ap %s' % posttrial])




pre_4ap_zscores = []
post_4ap_zscores = []
for key in allopticalResults.stim_responses_zscores.keys():
    count = 0
    for i in allopticalResults.pre_4ap_trials:
        if key in i[0]:
            count += 1

    for comp in range(count):
        comp += 1
        pre_4ap_df = allopticalResults.stim_responses_zscores[key][str(comp)]['pre-4ap']
        post_4ap_df = allopticalResults.stim_responses_zscores[key][str(comp)]['post-4ap']
        for col in pre_4ap_df.columns:
            if 'z' in str(col):
                pre_4ap_zscores = pre_4ap_zscores + list(pre_4ap_df[col][:-2])

        if len(post_4ap_df) > 0:
            for col in post_4ap_df.columns:
                if 'z' in str(col):
                    post_4ap_zscores = post_4ap_zscores + list(post_4ap_df[col][:-2])


data = [pre_4ap_zscores, post_4ap_zscores]
pj.plot_hist_density(data, x_label='z-score', title='All exps. stim responses zscores (normalized to pre-4ap) - pre vs. post',
                     fill_color=['green', 'purple'], num_bins=500,
                     figsize=(5, 4), legend_labels=['pre-4ap', 'post-4ap'], x_lim=[-15, 15])





# %% zscore of stim responses vs. TIME to seizure onset
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
    closest_sz_onset = pj.findClosest(list=expobj.seizure_lfp_onsets, input=stim_frame)[0]
    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
    stims_relative_sz.append(round(time_diff, 3))

cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x=post_4ap_df_zscore_stim_relative_to_sz.index, y=post_4ap_df_zscore_stim_relative_to_sz['avg'])
fig.show()