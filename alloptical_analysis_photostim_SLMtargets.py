## script dedicated to code that focuses on analysis re: SLM targets data

# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys; import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj

from skimage import draw

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-17'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013')

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


# %% 0) #### -------------------- ALL OPTICAL PHOTOSTIM ANALYSIS ################################################

# specify trials to run code on
code_run_list_all = []
for i in ['pre', 'post']:
    for key in list(allopticalResults.trial_maps[i].keys()):
        for j in range(len(allopticalResults.trial_maps[i][key])):
            code_run_list_all.append((i, key, j))

code_run_list_pre = []
for key in list(allopticalResults.trial_maps['pre'].keys()):
    for j in range(len(allopticalResults.trial_maps['pre'][key])):
        code_run_list_pre.append(('pre', key, j))

code_run_list_post4ap = []
for key in list(allopticalResults.trial_maps['post'].keys()):
    for j in range(len(allopticalResults.trial_maps['post'][key])):
        code_run_list_post4ap.append(('post', key, j))


short_list_pre = [('pre', 'e', '0')]
short_list_post = [('post', 'e', '0')]

# %% 1) adding slm targets responses to alloptical results allopticalResults.slmtargets_stim_responses

"""Not sure if this code is actually necessary..."""

animal_prep = 'PS07'
date = '2021-01-19'
# trial = 't-009'

pre4ap_trials = ['t-007', 't-008', 't-009']
post4ap_trials = ['t-011', 't-016', 't-017']

# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
#     date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
#
# expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

counter = allopticalResults.slmtargets_stim_responses.shape[0] + 1
# counter = 6

for trial in pre4ap_trials + post4ap_trials:
    print(counter)
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
        date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object

    expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

    # add trials info to experiment
    expobj.metainfo['pre4ap_trials'] = pre4ap_trials
    expobj.metainfo['post4ap_trials'] = post4ap_trials
    expobj.save()

    # save to results object:
    allopticalResults.slmtargets_stim_responses.loc[counter, 'prep_trial'] = '%s %s' % (
        expobj.metainfo['animal prep.'], expobj.metainfo['trial'])
    allopticalResults.slmtargets_stim_responses.loc[counter, 'date'] = expobj.metainfo['date']
    allopticalResults.slmtargets_stim_responses.loc[counter, 'exptype'] = expobj.metainfo['exptype']
    if 'post' in expobj.metainfo['exptype']:
        if hasattr(expobj, 'stims_in_sz'):
            allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean(
                [[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
            allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(
                list(expobj.outsz_StimSuccessRate_SLMtargets.values()))
        else:
            if not hasattr(expobj, 'seizure_lfp_onsets'):
                raise AttributeError(
                    'stims have not been classified as in or out of sz, no seizure lfp onsets for this trial')
            else:
                raise AttributeError(
                    'stims have not been classified as in or out of sz, but seizure lfp onsets attr was found, so need to troubleshoot further')

    else:
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean(
            [[np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(
            list(expobj.StimSuccessRate_SLMtargets.values()))

    allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dFF all targets)'] = np.nan
    counter += 1

allopticalResults.save()
allopticalResults.slmtargets_stim_responses


# %% 2.0.0) DATA COLLECTION SLMTargets: organize SLMTargets stim responses - across all appropriate pre4ap, post4ap trial comparisons - responses are dF/prestimF
""" doing it in this way so that its easy to use in the response vs. stim times relative to seizure onset code (as this has already been coded up)"""

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
stim_responses_comparisons_dict = {}
for i in range(len(allopticalResults.pre_4ap_trials)):
    prep = allopticalResults.pre_4ap_trials[i][0][:-6]
    pre4aptrial = allopticalResults.pre_4ap_trials[i][0][-5:]
    date = list(allopticalResults.slmtargets_stim_responses.loc[
                allopticalResults.slmtargets_stim_responses['prep_trial'] == '%s %s' % (
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
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'|-- importing {prep} {pre4aptrial_} - pre4ap trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False)
                    df_ = expobj.responses_SLMtargets.T

                    # append additional dataframe to the first dataframe
                    df.append(df_, ignore_index=True)
                else:
                    print(f"\-- ***** skipping: {prep} {pre4aptrial_}")

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in stim_responses_comparisons_dict.keys():
            stim_responses_comparisons_dict[prep] = {}
            comparison_number = 1
        else:
                comparison_number = len(stim_responses_comparisons_dict[prep]) + 1

        stim_responses_comparisons_dict[prep][f'{comparison_number}'] = {'pre-4ap': {}}
        stim_responses_comparisons_dict[prep][f'{comparison_number}']['pre-4ap'] = df

        pre_4ap_df = df


    else:
        print(f"|-- skipping: {prep} {pre4aptrial}")


    ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS
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

            stim_responses_comparisons_dict[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** 1 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))



    ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells
    # post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:] -- same as post4ap OUTSZ for loop one above



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

                stim_responses_comparisons_dict[prep][f"{comparison_number}"]['in sz'] = df
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
    stim_responses_comparisons_dict[prep][new_key]= stim_responses_comparisons_dict[prep].pop(f'{comparison_number}')
    # stim_responses_comparisons_dict[prep][new_key]= stim_responses_comparisons_dict[prep][f'{comparison_number}']


# save to: allopticalResults.stim_responses
allopticalResults.stim_responses = stim_responses_comparisons_dict
allopticalResults.save()

# %% 2.0.1) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: dF/prestimF - for loop over all experiments to collect responses in terms of sz onset time

stim_relative_szonset_vs_avg_response_alltargets_atstim = {}

for prep in allopticalResults.stim_responses.keys():
    # prep = 'PS07'

    for key in list(allopticalResults.stim_responses[prep].keys()):
        # key = list(allopticalResults.stim_responses[prep].keys())[0]
        # comp = 2
        if 'post-4ap' in allopticalResults.stim_responses[prep][key]:
            post_4ap_df = allopticalResults.stim_responses[prep][key]['post-4ap']
            if len(post_4ap_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, post4ap trial: {post4aptrial}')
                stim_relative_szonset_vs_avg_response_alltargets_atstim[f"{prep} {post4aptrial}"] = [[], []]
                expobj, experiment = aoutils.import_expobj(trial=post4aptrial, prep=prep, verbose=False)

                # transform the rows of the stims responses dataframe to relative time to seizure
                stims = list(post_4ap_df.index)
                stims_relative_sz = []
                for stim_idx in stims:
                    stim_frame = expobj.stim_start_frames[stim_idx]
                    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
                    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
                    stims_relative_sz.append(round(time_diff, 3))

                cols = [col for col in post_4ap_df.columns]
                post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
                post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

                # take average of all targets at a specific time to seizure onset
                post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

                stim_relative_szonset_vs_avg_response_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(stims_relative_sz)
                stim_relative_szonset_vs_avg_response_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(post_4ap_df_zscore_stim_relative_to_sz['avg'].tolist())


allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim = stim_relative_szonset_vs_avg_response_alltargets_atstim
allopticalResults.save()


# %% 2.1.0-dc) DATA COLLECTION SLMTargets: organize SLMTargets stim responses - across all appropriate pre4ap, post4ap trial comparisons - using whole trace dFF responses
""" doing it in this way so that its easy to use in the response vs. stim times relative to seizure onset code (as this has already been coded up)"""

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet.. just updated the sz onset/offset's
]

trials_run = [
    'PS11 t-010'
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
stim_responses_tracedFF_comparisons_dict = {}
for i in range(len(allopticalResults.pre_4ap_trials)):
    prep = allopticalResults.pre_4ap_trials[i][0][:-6]
    pre4aptrial = allopticalResults.pre_4ap_trials[i][0][-5:]
    post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:]
    date = allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == f"{prep} {pre4aptrial}", 'date'].values[0]
    print("\n\n\n-------------------------------------------")
    print(f"{i}, {date}, {prep}, pre4ap trial: {pre4aptrial}, post4ap trial: {post4aptrial}")


    # skipping some trials that need fixing of the expobj
    if f"{prep} {pre4aptrial}" not in trials_skip:


        # load up pre-4ap trial
        print(f'|-- importing {prep} {pre4aptrial} - pre4ap trial')



        expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep, verbose=False, do_processing=False)
        # collect raw Flu data from SLM targets
        expobj.collect_traces_from_targets(force_redo=False)
        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)  # REVIEW PROGRESS: pre4ap seems to be working fine till here for trace_dFF processing

        df = expobj.responses_SLMtargets_tracedFF.T  # df == stim frame x cells (photostim targets)
        if len(allopticalResults.pre_4ap_trials[i]) > 1:
            for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
                print(f"\---- {i}, {j}")
                # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                prep = allopticalResults.pre_4ap_trials[i][j][:-6]
                pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
                if f"{prep} {pre4aptrial}" not in trials_skip:
                    print(f"\------ adding trial to this comparison: {pre4aptrial_} [1.0]")
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'\------ importing {prep} {pre4aptrial_} - pre4ap trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False, do_processing=False)
                    # collect raw Flu data from SLM targets
                    expobj.collect_traces_from_targets(force_redo=False)
                    aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

                    df_ = expobj.responses_SLMtargets_tracedFF.T

                    # append additional dataframe to the first dataframe
                    df.append(df_, ignore_index=True)
                else:
                    print(f"\------ ***** skipping: {prep} {pre4aptrial_}")

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in stim_responses_tracedFF_comparisons_dict.keys():
            stim_responses_tracedFF_comparisons_dict[prep] = {}
            comparison_number = 1
        else:
            comparison_number = len(stim_responses_tracedFF_comparisons_dict[prep]) + 1

        # stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}'] = {'pre-4ap': {}, 'post-4ap': {}, 'in sz': {}}  # initialize dict for saving responses
        stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}'] = {'pre-4ap': {}}  # initialize dict for saving responses
        stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['pre-4ap'] = df


        pre_4ap_df = df


    else:
        print(f"|-- skipping: {prep} pre4ap trial {pre4aptrial}")


    ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS
    print(f'TEST 1.1 - working on {prep}, post4ap trial {post4aptrial}')


    # skipping some trials that need fixing of the expobj
    if f"{prep} {post4aptrial}" not in trials_skip:


        # load up post-4ap trial and stim responses
        print(f'|-- importing {prep} {post4aptrial} - post4ap trial')
        expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False, do_processing=False)
        # collect raw Flu data from SLM targets
        expobj.collect_traces_from_targets(force_redo=False)
        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

        if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
            df = expobj.responses_SLMtargets_tracedFF_outsz.T

            if len(allopticalResults.post_4ap_trials[i]) > 1:
                for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
                    print(f"\---- {i}, {j}")
                    # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                    prep = allopticalResults.post_4ap_trials[i][j][:-6]
                    post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
                    if f"{prep} {post4aptrial_}" not in trials_skip:
                        print(f"\------ adding trial to this comparison: {post4aptrial} [1.1]")
                        date = list(allopticalResults.slmtargets_stim_responses_tracedFF.loc[
                                        allopticalResults.slmtargets_stim_responses_tracedFF['prep_trial'] == '%s %s' % (prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'\------ importing {prep} {post4aptrial_} - post4ap trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False, do_processing=False)
                        # collect raw Flu data from SLM targets
                        expobj.collect_traces_from_targets(force_redo=False)
                        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

                        if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
                            df_ = expobj.responses_SLMtargets_tracedFF_outsz.T
                            # append additional dataframe to the first dataframe
                            df.append(df_, ignore_index=True)
                        else:
                            print('\------ **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\---- ***** skipping: {prep} post4ap trial {post4aptrial_}")

            stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** need to run collecting outsz responses SLMtargets attr for %s %s **** [1]' % (post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))



        ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells
        # post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:] -- same as post4ap OUTSZ for loop one above



        # skipping some trials that need fixing of the expobj
        # if f"{prep} {post4aptrial}" not in trials_skip:
        #     print(f'TEST 1.2 - working on {prep} {post4aptrial}')

        # using the same skip statement as in the main for loop here

        # load up post-4ap trial and stim responses
        expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False, do_processing=False)
        # collect raw Flu data from SLM targets
        expobj.collect_traces_from_targets(force_redo=False)
        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

        if hasattr(expobj, 'slmtargets_szboundary_stim'):
            if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
                df = expobj.responses_SLMtargets_tracedFF_insz.T


                # switch to NA for stims for cells which are classified in the sz
                # collect stim responses with stims excluded as necessary
                for target in df.columns:
                    # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
                    for stim in list(expobj.slmtargets_szboundary_stim.keys()):
                        if target in expobj.slmtargets_szboundary_stim[stim]:
                            df.loc[expobj.stim_start_frames.index(stim)][target] = np.nan

                    # responses = [expobj.responses_SLMtargets_tracedFF_insz.loc[col][expobj.stim_start_frames.index(stim)] for stim in expobj.stims_in_sz if
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
                            date = list(allopticalResults.slmtargets_stim_responses_tracedFF.loc[
                                            allopticalResults.slmtargets_stim_responses_tracedFF['prep_trial'] == '%s %s' % (
                                                prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep, verbose=False, do_processing=False)
                            # collect raw Flu data from SLM targets
                            expobj.collect_traces_from_targets(force_redo=False)
                            aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

                            if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
                                df_ = expobj.responses_SLMtargets_tracedFF_insz.T

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
                                    '**** need to run collecting in sz responses SLMtargets attr for %s %s **** [4]' % (post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} post4ap trial {post4aptrial_}")

                stim_responses_tracedFF_comparisons_dict[prep][f"{comparison_number}"]['in sz'] = df
            else:
                print('**** need to run collecting insz responses SLMtargets attr for %s %s **** [4]' % (post4aptrial, prep))
                allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
        else:
            print(f"**** need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial} [5]")

    else:
        print(f"\-- ***** skipping: {prep} post4ap trial {post4aptrial}")
        if not hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
            print(f'\-- **** need to run collecting outsz responses SLMtargets attr for post4ap trial {post4aptrial}, {prep} **** [1]')

        if not hasattr(expobj, 'slmtargets_szboundary_stim'):
            print(f'**** need to run collecting insz responses SLMtargets attr for post4ap trial {post4aptrial}, {prep} **** [2]')
        if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
            print(f'**** need to run collecting in sz responses SLMtargets attr for post4ap trial {post4aptrial}, {prep} **** [3]')

    ## switch out the comparison_number to something more readable
    new_key = f"{pre4aptrial} vs. {post4aptrial}"
    stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep].pop(f'{comparison_number}')
    # stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']

    # save to: allopticalResults.stim_responses_tracedFF
    allopticalResults.stim_responses_tracedFF = stim_responses_tracedFF_comparisons_dict
    allopticalResults.save()




# %% 2.1.1) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: delta(dFF) from whole trace - for loop over all experiments to collect responses in terms of sz onset time

stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim = {}

for prep in allopticalResults.stim_responses_tracedFF.keys():
    # prep = 'PS07's

    for key in list(allopticalResults.stim_responses_tracedFF[prep].keys()):
        # key = list(allopticalResults.stim_responses_tracedFF[prep].keys())[0]
        # comp = 2
        if 'post-4ap' in allopticalResults.stim_responses_tracedFF[prep][key]:
            post_4ap_df = allopticalResults.stim_responses_tracedFF[prep][key]['post-4ap']
            if len(post_4ap_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, post4ap trial: {post4aptrial}')
                stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[f"{prep} {post4aptrial}"] = [[], []]
                expobj, experiment = aoutils.import_expobj(trial=post4aptrial, prep=prep, verbose=False)

                # transform the rows of the stims responses dataframe to relative time to seizure
                stims = list(post_4ap_df.index)
                stims_relative_sz = []
                for stim_idx in stims:
                    stim_frame = expobj.stim_start_frames[stim_idx]
                    closest_sz_onset = pj.findClosest(arr=expobj.seizure_lfp_onsets, input=stim_frame)[0]
                    time_diff = (closest_sz_onset - stim_frame) / expobj.fps  # time difference in seconds
                    stims_relative_sz.append(round(time_diff, 3))

                cols = [col for col in post_4ap_df.columns]
                post_4ap_df_zscore_stim_relative_to_sz = post_4ap_df[cols]
                post_4ap_df_zscore_stim_relative_to_sz.index = stims_relative_sz  # take the original zscored df and assign a new index where the col names are times relative to sz onset

                # take average of all targets at a specific time to seizure onset
                post_4ap_df_zscore_stim_relative_to_sz['avg'] = post_4ap_df_zscore_stim_relative_to_sz.T.mean()

                stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(stims_relative_sz)
                stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(post_4ap_df_zscore_stim_relative_to_sz['avg'].tolist())

    allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim = stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim
    print(f"length of allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim dict: {len(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())}")
    allopticalResults.save()

