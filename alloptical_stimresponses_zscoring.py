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


# %% convert stim responses to z-scores - relative to pre-4ap scores

# pre-4ap trial

prep = 'RL108'
trial = 't-010'
date = '2020-12-18'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)


df = expobj.responses_SLMtargets.T
cols = list(df.columns)
for col in cols:
    col_zscore = str(col) + '_z'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
# -- add a mean and std calculation for each cell to use for the post-4ap trial scores
mean = pd.Series(df.mean(), name='mean')
std = pd.Series(df.std(ddof=0), name='std')
df = df.append([mean, std])
expobj.responses_SLMtargets_zscore = df
expobj.save()

pre_4ap_df = df


# post-4ap trial - use the mean and std of the same SLM target calculated from the pre-4ap trial
trial = 't-011'
expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)

df = expobj.outsz_responses_SLMtargets.T
cols = list(df.columns)
for col in cols:
    col_zscore = str(col) + '_z'
    df[col_zscore] = (df[col] - pre_4ap_df.loc['mean', col])/pre_4ap_df.loc['std', col]

expobj.responses_SLMtargets_zscore = df
expobj.save()

post_4ap_df = df

# %% plot histogram of zscore stim responses pre and post 4ap
prep = 'RL108'
date = '2020-12-18'

# post 4ap df
posttrial = 't-011'
expobj, experiment = aoutils.import_expobj(trial=posttrial, date=date, prep=prep)
post_4ap_df = expobj.responses_SLMtargets_zscore

# pre 4ap df
pretrial = 't-010'
expobj, experiment = aoutils.import_expobj(trial=pretrial, date=date, prep=prep)
pre_4ap_df = expobj.responses_SLMtargets_zscore


pre_4ap_zscores = []
for col in pre_4ap_df.columns:
    if 'z' in str(col):
        pre_4ap_zscores = pre_4ap_zscores + list(pre_4ap_df[col][:-2])

post_4ap_zscores = []
for col in post_4ap_df.columns:
    if 'z' in str(col):
        post_4ap_zscores = post_4ap_zscores + list(post_4ap_df[col][:-2])

data = [pre_4ap_zscores, post_4ap_zscores]
pj.plot_hist_density(data, x_label='z-score', title='stim zscores (normalized to pre-4ap) - pre vs. post', fill_color=['blue', 'orange'],
                     figsize=(5,4), legend_labels=['pre-4ap %s' % pretrial, 'post-4ap %s' % posttrial])

# %% zscore of stims vs. time to seizure onset
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
    time_diff = (closest_sz_onset - stim_frame) / expobj.fps
    stims_relative_sz.append(round(time_diff, 3))

cols = [col for col in post_4ap_df.columns if 'z' in str(col)]
post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz

post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x=post_4ap_df_zscore_stim_relative_to_sz.index, y=post_4ap_df_zscore_stim_relative_to_sz['avg'])
fig.show()