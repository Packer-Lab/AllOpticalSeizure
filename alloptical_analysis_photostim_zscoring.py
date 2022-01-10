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



# %% 1.1) DATA COLLECTION: dfprestimf; organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons

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
                for j, _ in enumerate(allopticalResults.post_4ap_trials[i]):
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
    # if f"{prep} {post4aptrial}" not in skip_trials:
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


# %% 1.2) DATA COLLECTION: delta(trace_dFF); organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons

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
    if f"{prep} {pre4aptrial}"  not in trials_skip:
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
            if type(expobj.responses_SLMtargets_tracedFF_outsz) is list:
                expobj.StimSuccessRate_SLMtargets_tracedFF_outsz, expobj.hits_SLMtargets_tracedFF_outsz, expobj.responses_SLMtargets_tracedFF_outsz, expobj.traces_SLMtargets_tracedFF_successes_outsz = \
                    expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10,
                                                       stims_to_use=expobj.stims_out_sz)

            response_df = expobj.responses_SLMtargets_tracedFF_outsz.T

            if len(allopticalResults.post_4ap_trials[i]) > 1:
                for j, _ in enumerate(allopticalResults.post_4ap_trials[i]):
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
    # if f"{prep} {post4aptrial}" not in skip_trials:
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
                    for j, _ in enumerate(allopticalResults.post_4ap_trials[i]):
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


        ## switch out this comparison_number to something more readable
        new_key = f"{pre4aptrial} vs. {post4aptrial}"
        allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][new_key] = allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep].pop(f'{comparison_number}')
        # allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][new_key]= allopticalResults.stim_responses_zscores['delta(trace_dFF)'][prep][f'{comparison_number}']

    else:
        print(f"\-- ***** skipping: {prep} {post4aptrial}")
        # if not hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
        #     print(f'\-- **** 1 need to run collecting outsz responses SLMtargets attr for {post4aptrial}, {prep} ****')
        #
        # if not hasattr(expobj, 'slmtargets_szboundary_stim'):
        #     print(f'**** 2 need to run collecting insz responses SLMtargets attr for {post4aptrial}, {prep} ****')
        # if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
        #     print(f'**** 3 need to run collecting in sz responses SLMtargets attr for {post4aptrial}, {prep} ****')


allopticalResults.save()


# %% 1.2.1-dc) DATA COLLECTION: delta(trace_dFF); organize and convert SLMTargets stim responses to Z-SCORES - relative to pre-4ap scores - make sure to compare the appropriate pre and post 4ap trial comparisons


post4ap_expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-017')


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, skip_trials=['PS05 t-012'])
def zscoring_post4apResponses_to_pre4apResponses(to_save, **kwargs):

    post4ap_expobj = kwargs['expobj']
    prep = post4ap_expobj.prep
    print(f"\- zscoring post4ap responses to pre4ap responses for: {post4ap_expobj.t_series_name}")
    for key in [*allopticalResults.trial_maps['post']]:
        if post4ap_expobj.t_series_name in allopticalResults.trial_maps['post'][key]:
            break

    # zscoring pre4ap responses
    pre4aptrial = allopticalResults.trial_maps['pre'][key][0][-5:]  # first trial (if there is multi trials in this comparison)
    pre4ap_expobj, _ = aoutils.import_expobj(prep=prep, trial=pre4aptrial)
    response_df = pre4ap_expobj.responses_SLMtargets_tracedFF.T

    if len(allopticalResults.trial_maps['pre'][key]) > 1:
        for t_series_name in allopticalResults.trial_maps['pre'][key][1:]:
            pre4aptrial = t_series_name[-5:]  # first trial (if there is multi trials in this comparison)
            pre4ap_expobj, _ = aoutils.import_expobj(prep=prep, trial=pre4aptrial)
            response_df_ = pre4ap_expobj.responses_SLMtargets_tracedFF.T
            # append additional dataframe to the first dataframe
            response_df = response_df.append(response_df_, ignore_index=True)

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
    if prep not in [*to_save]:
        to_save[prep] = {}
        comparison_number = 1
    else:
        comparison_number = len(to_save[prep]) + 1

    to_save[prep][f'{comparison_number}'] = {}
    to_save[prep][f'{comparison_number}']['pre-4ap'] = response_df

    pre_4ap_response_df = response_df


    # zscoring post4ap responses to pre4ap responses
    response_df = post4ap_expobj.responses_SLMtargets_tracedFF.T

    if len(allopticalResults.trial_maps['post'][key]) > 1:
        for t_series_name in allopticalResults.trial_maps['post'][key][1:]:
            post4aptrial = t_series_name[-5:]  # first trial (if there is multi trials in this comparison)
            post4ap_expobj, _ = aoutils.import_expobj(prep=prep, trial=post4aptrial)
            response_df_ = post4ap_expobj.responses_SLMtargets_tracedFF.T
            # append additional dataframe to the first dataframe
            response_df = response_df.append(response_df_, ignore_index=True)

    cols = list(response_df.columns)
    # for loop for z scoring all stim responses for all cells - creates a whole set of new columns for each cell
    # NOTE THAT THE Z SCORING IS BEING DONE RELATIVE TO THE MEAN AND STD OF THE SAME TARGET FROM THE PRE4AP response_df
    for col in cols:
        col_zscore = str(col) + '_z'
        response_df[col_zscore] = (response_df[col] - pre_4ap_response_df.loc['mean', col]) / \
                                  pre_4ap_response_df.loc['std', col]

    to_save[prep][f'{comparison_number}']['post-4ap'] = response_df

    print(f"|- keys in to_save arg: {[*to_save]} [1.2.1-2]")
    # to_save

to_save = {}
zscoring_post4apResponses_to_pre4apResponses(to_save=to_save)
allopticalResults.stim_responses_zscores['delta(trace_dFF)'] = to_save

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



# %% 3.0-dc) DATA COLLECTION - zscore of stim responses vs. TIME to seizure onset - for loop over all experiments to collect zscores in terms of sz onset time

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

                # transform the rows of the stims responses dataframe to relative TIME to seizure
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


# %% 4.0-dc) DATA COLLECTION - zscore of stim responses vs. distance to seizure wavefront

expobj, experiment = aoutils.import_expobj(prep='RL108', trial='t-011')


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def tempfunc(**kwargs):
    expobj = kwargs['expobj']

    if not hasattr(expobj, 'distance_to_sz'):
        expobj.distance_to_sz = {'SLM Targets': ['uninitialized'],
                               's2p nontargets': ['uninitialized']}  # calculating the distance between the sz wavefront and cells

    if 'uninitialized' in expobj.distance_to_sz['SLM Targets']:
        expobj.sz_locations_stims()
        print(f'** WARNING: rerunning {expobj.t_series_name} .calcMinDistanceToSz [4.0-1]')
        x_ = expobj.calcMinDistanceToSz()
        expobj.save()

tempfunc()


# key = 'l'; exp = 'post'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")
# expobj, experiment = aoutils.import_expobj(prep='PS04', trial='t-018')



@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def collect_responses_vs_distance_to_seizure_SLMTargets_pre4ap_zscored(**kwargs):
    print(f"\t\- collecting pre4ap zscored responses vs. distance to seizure [4.0-2]")
    expobj = kwargs['expobj']

    # transform the rows of the stims responses dataframe to relative DISTANCE to seizure
    trials_comparison = [*allopticalResults.stim_responses_zscores['delta(trace_dFF)'][expobj.prep]][0]

    # trials_comparison = [*allopticalResults.stim_responses_zscores[expobj.prep]][0]
    post_4ap_response_df = allopticalResults.stim_responses_zscores['delta(trace_dFF)'][expobj.prep][trials_comparison]['post-4ap']

    # (re-)make pandas dataframe
    df = pd.DataFrame(columns=['target_id', 'stim_id', 'inorout_sz', 'distance_to_sz', 'response (z-scored to pre4ap)'])
    cols = [col for col in post_4ap_response_df.columns if 'z' in str(col)]
    post_4ap_response_df_zscore_stim_relative_to_sz = post_4ap_response_df[cols]
    # post_4ap_response_df_zscore_stim_relative_to_sz.columns

    stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
    for target in range(expobj.n_targets_total):
        # idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        for stim_idx, stim in stim_ids:
            if target in expobj.slmtargets_szboundary_stim[stim]: inorout_sz = 'in'
            else: inorout_sz = 'out'

            distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]
            response_zscored_pre4ap = post_4ap_response_df_zscore_stim_relative_to_sz.loc[stim_idx, f"{target}_z"]

            df = df.append({'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                            'response (z-scored to pre4ap)': response_zscored_pre4ap}, ignore_index=True)

    expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets = df

    # convert distances to microns
    expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets['distance_to_sz_um'] = round(expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets['distance_to_sz'] / expobj.pix_sz_x, 2)
    expobj.save()


collect_responses_vs_distance_to_seizure_SLMTargets_pre4ap_zscored()

# %% 4.1-dc) PLOTTING - collect and plot targets responses for stims vs. distance

expobj, experiment = aoutils.import_expobj(prep='RL108', trial='t-013')

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, skip_trials=['PS05 t-012'])
def plot_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    # response_type = 'dFF (z scored)'

    print(f"\t|- plotting responses vs. distance to seizure [5.1-1]")
    expobj = kwargs['expobj']
    response_type = 'response (z-scored to pre4ap)'
    fig, ax = plt.subplots(figsize=[8, 3])
    for target in range(expobj.n_targets_total):
        target = 0
        idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        indexes = expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets[expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets['target_id'] == target].index
        responses = expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets.loc[indexes, response_type]
        distance_to_sz = expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets.loc[indexes, 'distance_to_sz_um']

        positive_distances = np.where(distance_to_sz > 0)
        negative_distances = np.where(distance_to_sz < 0)

        pj.make_general_scatter(x_list=[distance_to_sz[positive_distances]], y_data=[responses[positive_distances]], fig=fig, ax=ax, colors=['cornflowerblue'], alpha=0.5, s=30, show=False,
                                x_label='distance to sz', y_label=response_type)
        pj.make_general_scatter(x_list=[distance_to_sz[negative_distances]], y_data=[responses[negative_distances]], fig=fig, ax=ax, colors=['tomato'], alpha=0.5, s=30, show=False,
                                x_label='distance to sz', y_label=response_type)

    fig.suptitle(expobj.t_series_name)
    fig.show()


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, skip_trials=['PS05 t-012'])
def plot_collection_response_distance(**kwargs):
    print(f"\t|- plotting a collection of plots measuring responses vs. distance to seizure [5.1-2]")
    expobj = kwargs['expobj']
    response_type = 'response (z-scored to pre4ap)'
    if not hasattr(expobj, 'responses_SLMtargets_tracedFF_avg_df'):
        expobj.avgResponseSzStims_SLMtargets(save=True)

    data = expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets
    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=[18, 4])
    axs[0] = sns.boxplot(data=expobj.responses_SLMtargets_tracedFF_avg_df, x='stim_group', y='avg targets response', order=['interictal', 'ictal'],
                         width=0.5, ax=axs[0], palette=['tomato', 'mediumseagreen'])  # plotting mean across stims (len= # of targets)
    axs[0] = sns.swarmplot(data=expobj.responses_SLMtargets_tracedFF_avg_df, x='stim_group', y='avg targets response', order=['interictal', 'ictal'],
                           color=".25", ax=axs[0])
    sns.stripplot(x="inorout_sz", y="distance_to_sz_um", data=data, ax=axs[1], alpha=0.2, order=['in', 'out'])
    axs[2] = sns.violinplot(x="inorout_sz", y=response_type, data=data, legend=False, ax=axs[2], order=['in', 'out'])
    axs[2].set_ylim([-3, 3])
    axs[3] = sns.scatterplot(data=data, x='distance_to_sz_um', y=response_type, ax=axs[3], alpha=0.2, hue='distance_to_sz_um', hue_norm=(-1,1),
                             palette=sns.diverging_palette(240, 10, as_cmap=True), legend=False)
    axs[3].set_ylim([-3, 3])
    aoplot.plot_sz_boundary_location(expobj, fig=fig, ax=axs[4], title=None)
    fig.suptitle(f"{expobj.t_series_name} - {response_type}")
    fig.tight_layout(pad=1.1)
    fig.show()

plot_responses_vs_distance_to_seizure_SLMTargets()
plot_collection_response_distance()



# %% 4.1.1-dc) binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses - represent the distances in percentile space

# expobj, experiment = aoutils.import_expobj(prep='RL108', trial='t-013')

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_responsesPre4apZscored_vs_distance_to_seizure_SLMTargets_2ddensity(positive_distances_only = False, plot=True, **kwargs):

    print(f"\t|- plotting responses vs. distance to seizure")
    expobj = kwargs['expobj']

    response_type = 'response (z-scored to pre4ap)'

    data_expobj = np.array([[], []]).T
    for target in expobj.responses_SLMtargets_tracedFF.index:
        idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        indexes = expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets[expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets['target_id'] == target].index
        responses = np.asarray(expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets.loc[indexes, response_type])
        distance_to_sz = np.asarray(expobj.responsesPre4apZscored_vs_distance_to_seizure_SLMTargets.loc[indexes, 'distance_to_sz_um'])


        if positive_distances_only:
            distance_to_sz_pos = np.where(distance_to_sz > 0)[0]
            responses_posdistances = responses[distance_to_sz_pos]

            _data = np.array([distance_to_sz_pos, responses_posdistances]).T
        else:
            _data = np.array([distance_to_sz, responses]).T

        data_expobj = np.vstack((_data, data_expobj))



    distances_to_sz = data_expobj[:, 0]
    bin_size = 20  # um
    # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
    bins_num = 40

    pj.plot_hist2d(data=data_expobj, bins=[100,100], y_label=response_type, title=expobj.t_series_name, figsize=(4, 2), x_label='distance to seizure (um)',
                   y_lim=[-2,2]) if plot else None


    return data_expobj

data = plot_responsesPre4apZscored_vs_distance_to_seizure_SLMTargets_2ddensity(positive_distances_only = False, plot=True)

def convert_responsesPre4apZscored_szdistances_percentile_space():
    data_all = np.array([[], []]).T
    for data_ in data:
        data_all = np.vstack((data_, data_all))

    from scipy.stats import percentileofscore

    distances_to_sz = data_all[:, 0]
    idx_sorted = np.argsort(distances_to_sz)
    distances_to_sz_sorted = distances_to_sz[idx_sorted]
    responses_sorted = data_all[:, 1][idx_sorted]
    s = pd.Series(distances_to_sz_sorted)
    percentiles = s.apply(lambda x: percentileofscore(distances_to_sz_sorted, x))
    scale_percentile_distances = {}
    for pct in range(0, 100):
        scale_percentile_distances[int(pct+1)] = np.round(np.percentile(distances_to_sz_sorted, pct),0)
    data_all = np.array([percentiles, responses_sorted]).T

    return data_all, percentiles, responses_sorted, distances_to_sz_sorted, scale_percentile_distances

data_all, percentiles, responses_sorted, distances_to_sz_sorted, scale_percentile_distances = convert_responsesPre4apZscored_szdistances_percentile_space()

def plot_density_responsesPre4apZscored_szdistances(data_all=data_all, distances_to_sz_sorted=distances_to_sz_sorted):
    # plotting density plot for all exps, in percentile space (to normalize for excess of data at distances which are closer to zero) - TODO any smoothing?

    response_type = 'response (z-scored to pre4ap)'


    bin_size = 5  # um
    # bins_num = int((max(distances_to_sz) - min(distances_to_sz)) / bin_size)
    bins_num = [100, 500]

    fig, ax = plt.subplots(figsize=(6,3))
    pj.plot_hist2d(data=data_all, bins=bins_num, y_label=response_type, figsize=(6, 3), x_label='distance to seizure (%tile space)',
                   title=f"2d density plot, all exps, 50%tile = {np.percentile(distances_to_sz_sorted, 50)}um",
                   y_lim=[-3, 3], fig=fig, ax=ax, show=False)
    ax.axhline(0, ls='--', c='white', lw=1)
    xticks = [1, 25, 50, 57, 75, 100]  # percentile space
    ax.set_xticks(ticks=xticks)
    labels = [scale_percentile_distances[x_] for x_ in xticks]
    ax.set_xticklabels(labels)
    ax.set_xlabel('distance to seizure (um)')

    fig.show()

plot_density_responsesPre4apZscored_szdistances()



# plotting line plot for all datapoints for responses vs. distance to seizure

def plot_lineplot_responsesPre4apZscored_pctszdistances(percentiles, responses_sorted):
    response_type = 'response (z-scored to pre4ap)'

    percentiles_binned = np.round(percentiles)

    bin = 5
    # change to pct% binning
    percentiles_binned = (percentiles_binned // bin) * bin

    d = {'distance to seizure (%tile space)': percentiles_binned,
         response_type: responses_sorted}

    df = pd.DataFrame(d)

    fig, ax = plt.subplots(figsize=(6,3))
    sns.lineplot(data=df, x='distance to seizure (%tile space)', y=response_type, ax=ax)
    ax.set_title(f'responses over distance to sz, all exps, normalized to percentile space ({bin}% bins)', wrap=True)
    ax.margins(0.02)
    ax.axhline(0, ls='--', c='orange', lw=1)

    xticks = [1, 25, 50, 57, 75, 100]  # percentile space
    ax.set_xticks(ticks=xticks)
    labels = [scale_percentile_distances[x_] for x_ in xticks]
    ax.set_xticklabels(labels)
    ax.set_xlabel('distance to seizure (um)')

    fig.tight_layout(pad=2)
    plt.show()

plot_lineplot_responsesPre4apZscored_pctszdistances(percentiles, responses_sorted)
