#### FILE FOR PUTTING TOGEHTER CODE TO RUN ON THE SERVER

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj


# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-23'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


# expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013')

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

# %% troubleshoot

# prep_trial = 'PS11 t-011'
#
# expobj, _ = aoutils.import_expobj(exp_prep=prep_trial)
#
# print(expobj.responses_SLMtargets_tracedFF_insz)
# print(expobj.responses_SLMtargets_tracedFF_outsz)


### need to re run alloptical processing photostim

# @aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False)
# def run_on_server(**kwargs):
#     expobj = kwargs['expobj']
#     print(expobj.metainfo)
#
#
# run_on_server()



# ### need to re run alloptical processing photostim
#
# @aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
# def run_on_server(**kwargs):
#     expobj = kwargs['expobj']
#     # temp run once fully
#     aoutils.run_alloptical_processing_photostim(expobj, plots=False,
#                                                 force_redo=False)
#
# run_on_server()


# aoutils.run_alloptical_processing_photostim(expobj, plots=False)

# %% aoanalysis-photostim-zscoring-1.1) DATA COLLECTION: dfprestimf; organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
allopticalResults.stim_responses_zscores['dfprestimf'] = {}
for i, _ in enumerate(allopticalResults.pre_4ap_trials):
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

        response_df = expobj.responses_SLMtargets_dfprestimf.T  # df == stim frame x cells (photostim targets)
        if len(allopticalResults.pre_4ap_trials[i]) > 1:
            for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
                print(f"|-- {i}, {j}")
                # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                prep = allopticalResults.pre_4ap_trials[i][j][:-6]
                pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
                if f"{prep} {pre4aptrial}" not in trials_skip:
                    print(f"adding trial to this comparison: {pre4aptrial_} [1.0]")
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'|-- importing {prep} {pre4aptrial_} - pre4ap trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False)
                    response_df_ = expobj.responses_SLMtargets_dfprestimf.T

                    # append additional dataframe to the first dataframe
                    response_df.append(response_df_, ignore_index=True)
                else:
                    print(f"\-- ***** skipping: {prep} {pre4aptrial_}")

        cols = list(response_df.columns)  # cols = cells
        # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
        for col in cols:
            col_zscore = str(col) + '_z'
            response_df[col_zscore] = (response_df[col] - response_df[col].mean())/response_df[col].std(ddof=0)
        # -- add a mean and std calculation for each cell to use for the post-4ap trial scores
        mean = pd.Series(response_df.mean(), name='mean')
        std = pd.Series(response_df.std(ddof=0), name='std')
        response_df = response_df.append([mean, std])

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in allopticalResults.stim_responses_zscores['dfprestimf'].keys():
            allopticalResults.stim_responses_zscores['dfprestimf'][prep] = {}
            comparison_number = 1
        else:
            comparison_number = len(allopticalResults.stim_responses_zscores['dfprestimf'][prep]) + 1

        allopticalResults.stim_responses_zscores['dfprestimf'][prep][f'{comparison_number}'] = {}
        allopticalResults.stim_responses_zscores['dfprestimf'][prep][f'{comparison_number}']['pre-4ap'] = response_df

        # allopticalResults.save()


        # expobj.responses_SLMtargets_zscore = df
        # expobj.save()

        pre_4ap_response_df = response_df


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
        if hasattr(expobj, 'responses_SLMtargets_dfprestimf_outsz'):
            response_df = expobj.responses_SLMtargets_dfprestimf_outsz.T

            if len(allopticalResults.post_4ap_trials[i]) > 1:
                for j, _ in enumerate(allopticalResults.post_4ap_trials[i])[1:]:
                    print(f"|-- {i}, {j}")
                    # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                    prep = allopticalResults.post_4ap_trials[i][j][:-6]
                    post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                    if f"{prep} {post4aptrial_}" not in trials_skip:
                        print(f"adding trial to this comparison: {post4aptrial} [1.1]")
                        date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'|-- importing {prep} {post4aptrial_} - post4ap trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                        if hasattr(expobj, 'responses_SLMtargets_dfprestimf_outsz'):
                            response_df_ = expobj.responses_SLMtargets_dfprestimf_outsz.T
                            # append additional dataframe to the first dataframe
                            response_df.append(response_df_, ignore_index=True)
                        else:
                            print('|-- **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\-- ***** skipping: {prep} {post4aptrial_}")

            cols = list(response_df.columns)
            # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
            # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP response_df
            for col in cols:
                col_zscore = str(col) + '_z'
                response_df[col_zscore] = (response_df[col] - pre_4ap_response_df.loc['mean', col])/pre_4ap_response_df.loc['std', col]

            allopticalResults.stim_responses_zscores['dfprestimf'][prep][f'{comparison_number}']['post-4ap'] = response_df

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
            if hasattr(expobj, 'responses_SLMtargets_dfprestimf_insz'):
                response_df = expobj.responses_SLMtargets_dfprestimf_insz.T


                # switch to NA for stims for cells which are classified in the sz
                # collect stim responses with stims excluded as necessary
                for target in response_df.columns:
                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                        if target in expobj.slmtargets_szboundary_stim[stim]:
                            response_df.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                    # responses = [expobj.responses_SLMtargets_dfprestimf_insz.loc[col][expobj.stim_start_frames.index(stim)] for stim in expobj.stims_in_sz if
                    #              col not in expobj.slmtargets_szboundary_stim[stim]]
                    # targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)


                if len(allopticalResults.post_4ap_trials[i]) > 1:
                    for j, _ in enumerate(allopticalResults.post_4ap_trials[i])[1:]:
                        print(f"|-- {i}, {j}")
                        # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                        prep = allopticalResults.post_4ap_trials[i][j][:-6]
                        post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                        if f"{prep} {post4aptrial_}" not in trials_skip:
                            print(f"{post4aptrial} [1.2]")
                            date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                                prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                            if hasattr(expobj, 'responses_SLMtargets_dfprestimf_insz'):
                                response_df_ = expobj.responses_SLMtargets_dfprestimf_insz.T

                                # append additional dataframe to the first dataframe
                                # switch to NA for stims for cells which are classified in the sz
                                # collect stim responses with stims excluded as necessary
                                for target in response_df.columns:
                                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                                        if target in expobj.slmtargets_szboundary_stim[stim]:
                                            response_df_.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                                response_df.append(response_df_, ignore_index=True)
                            else:
                                print(
                                    '**** 4 need to run collecting in sz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} {post4aptrial_}")

                cols = list(response_df.columns)
                # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
                # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP response_df
                for col in cols:
                    col_zscore = str(col) + '_z'
                    response_df[col_zscore] = (response_df[col] - pre_4ap_response_df.loc['mean', col]) / pre_4ap_response_df.loc['std', col]

                allopticalResults.stim_responses_zscores['dfprestimf'][prep][f"{comparison_number}"]['in sz'] = response_df
            else:
                print('**** 4 need to run collecting insz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
                allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
        else:
            print(f"**** 5 need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial}")

    else:
        print(f"\-- ***** skipping: {prep} {post4aptrial}")
        if not hasattr(expobj, 'responses_SLMtargets_dfprestimf_outsz'):
            print(f'\-- **** 1 need to run collecting outsz responses SLMtargets attr for {post4aptrial}, {prep} ****')

        if not hasattr(expobj, 'slmtargets_szboundary_stim'):
            print(f'**** 2 need to run collecting insz responses SLMtargets attr for {post4aptrial}, {prep} ****')
        if hasattr(expobj, 'responses_SLMtargets_dfprestimf_insz'):
            print(f'**** 3 need to run collecting in sz responses SLMtargets attr for {post4aptrial}, {prep} ****')

    ## switch out this comparison_number to something more readable
    new_key = f"{pre4aptrial} vs. {post4aptrial}"
    allopticalResults.stim_responses_zscores['dfprestimf'][prep][new_key] = allopticalResults.stim_responses_zscores['dfprestimf'][prep].pop(f'{comparison_number}')
    # allopticalResults.stim_responses_zscores['dfprestimf'][prep][new_key]= allopticalResults.stim_responses_zscores['dfprestimf'][prep][f'{comparison_number}']


allopticalResults.save()


# %% aoanalysis-photostim-zscoring-1.2) DATA COLLECTION: delta(trace_dFF); organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
allopticalResults.stim_responses_zscores['delta(trace_dFF)'] = {}
for i, _ in enumerate(allopticalResults.pre_4ap_trials):
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

        response_df = expobj.responses_SLMtargets_tracedFF.T  # df == stim frame x cells (photostim targets)
        if len(allopticalResults.pre_4ap_trials[i]) > 1:
            for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
                print(f"|-- {i}, {j}")
                # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                prep = allopticalResults.pre_4ap_trials[i][j][:-6]
                pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
                if f"{prep} {pre4aptrial}" not in trials_skip:
                    print(f"adding trial to this comparison: {pre4aptrial_} [1.0]")
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'|-- importing {prep} {pre4aptrial_} - pre4ap trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False)
                    response_df_ = expobj.responses_SLMtargets_tracedFF.T

                    # append additional dataframe to the first dataframe
                    response_df.append(response_df_, ignore_index=True)
                else:
                    print(f"\-- ***** skipping: {prep} {pre4aptrial_}")

        cols = list(response_df.columns)  # cols = cells
        # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
        for col in cols:
            col_zscore = str(col) + '_z'
            response_df[col_zscore] = (response_df[col] - response_df[col].mean())/response_df[col].std(ddof=0)
        # -- add a mean and std calculation for each cell to use for the post-4ap trial scores
        mean = pd.Series(response_df.mean(), name='mean')
        std = pd.Series(response_df.std(ddof=0), name='std')
        response_df = response_df.append([mean, std])

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in allopticalResults.stim_responses_zscores['delta(trace_dFF)'].keys():
            allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep] = {}
            comparison_number = 1
        else:
            comparison_number = len(allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep]) + 1

        allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f'{comparison_number}'] = {}
        allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f'{comparison_number}']['pre-4ap'] = response_df

        # allopticalResults.save()


        # expobj.responses_SLMtargets_zscore = df
        # expobj.save()

        pre_4ap_response_df = response_df


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
        if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
            response_df = expobj.responses_SLMtargets_tracedFF_outsz.T

            if len(allopticalResults.post_4ap_trials[i]) > 1:
                for j, _ in enumerate(allopticalResults.post_4ap_trials[i])[1:]:
                    print(f"|-- {i}, {j}")
                    # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                    prep = allopticalResults.post_4ap_trials[i][j][:-6]
                    post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                    if f"{prep} {post4aptrial_}" not in trials_skip:
                        print(f"adding trial to this comparison: {post4aptrial} [1.1]")
                        date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'|-- importing {prep} {post4aptrial_} - post4ap trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                        if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
                            response_df_ = expobj.responses_SLMtargets_tracedFF_outsz.T
                            # append additional dataframe to the first dataframe
                            response_df.append(response_df_, ignore_index=True)
                        else:
                            print('|-- **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\-- ***** skipping: {prep} {post4aptrial_}")

            cols = list(response_df.columns)
            # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
            # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP response_df
            for col in cols:
                col_zscore = str(col) + '_z'
                response_df[col_zscore] = (response_df[col] - pre_4ap_response_df.loc['mean', col])/pre_4ap_response_df.loc['std', col]

            allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f'{comparison_number}']['post-4ap'] = response_df

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
            if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
                response_df = expobj.responses_SLMtargets_tracedFF_insz.T


                # switch to NA for stims for cells which are classified in the sz
                # collect stim responses with stims excluded as necessary
                for target in response_df.columns:
                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                        if target in expobj.slmtargets_szboundary_stim[stim]:
                            response_df.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                    # responses = [expobj.responses_SLMtargets_tracedFF_insz.loc[col][expobj.stim_start_frames.index(stim)] for stim in expobj.stims_in_sz if
                    #              col not in expobj.slmtargets_szboundary_stim[stim]]
                    # targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)


                if len(allopticalResults.post_4ap_trials[i]) > 1:
                    for j, _ in enumerate(allopticalResults.post_4ap_trials[i])[1:]:
                        print(f"|-- {i}, {j}")
                        # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                        prep = allopticalResults.post_4ap_trials[i][j][:-6]
                        post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                        if f"{prep} {post4aptrial_}" not in trials_skip:
                            print(f"{post4aptrial} [1.2]")
                            date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                                prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False)
                            if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
                                response_df_ = expobj.responses_SLMtargets_tracedFF_insz.T

                                # append additional dataframe to the first dataframe
                                # switch to NA for stims for cells which are classified in the sz
                                # collect stim responses with stims excluded as necessary
                                for target in response_df.columns:
                                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                                        if target in expobj.slmtargets_szboundary_stim[stim]:
                                            response_df_.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                                response_df.append(response_df_, ignore_index=True)
                            else:
                                print(
                                    '**** 4 need to run collecting in sz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} {post4aptrial_}")

                cols = list(response_df.columns)
                # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
                # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP response_df
                for col in cols:
                    col_zscore = str(col) + '_z'
                    response_df[col_zscore] = (response_df[col] - pre_4ap_response_df.loc['mean', col]) / pre_4ap_response_df.loc['std', col]

                allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f"{comparison_number}"]['in sz'] = response_df
            else:
                print('**** 4 need to run collecting insz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
                allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
        else:
            print(f"**** 5 need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial}")

    else:
        print(f"\-- ***** skipping: {prep} {post4aptrial}")
        if not hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
            print(f'\-- **** 1 need to run collecting outsz responses SLMtargets attr for {post4aptrial}, {prep} ****')

        if not hasattr(expobj, 'slmtargets_szboundary_stim'):
            print(f'**** 2 need to run collecting insz responses SLMtargets attr for {post4aptrial}, {prep} ****')
        if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
            print(f'**** 3 need to run collecting in sz responses SLMtargets attr for {post4aptrial}, {prep} ****')

    ## switch out this comparison_number to something more readable
    new_key = f"{pre4aptrial} vs. {post4aptrial}"
    allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][new_key] = allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep].pop(f'{comparison_number}')
    # allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][new_key]= allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f'{comparison_number}']


allopticalResults.save()





"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


sys.exit()
