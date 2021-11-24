#### FILE FOR PUTTING TOGEHTER CODE TO RUN ON THE SERVER

# IMPORT MODULES AND TRIAL expobj OBJECT
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


# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-20'
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

# %% aoanalysis-6.1.0-dc) DATA COLLECTION: organize SLMTargets stim responses - across all appropriate pre4ap, post4ap trial comparisons - using whole trace dFF responses
""" doing it in this way so that its easy to use in the response vs. stim times relative to seizure onset code (as this has already been coded up)"""

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
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
        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

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


# %% aoanalysis-6.1.1) DATA COLLECTION - absolute stim responses vs. TIME to seizure onset - responses: delta(dFF) from whole trace - for loop over all experiments to collect responses in terms of sz onset time

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
                    closest_sz_onset = pj.findClosest(ls=expobj.seizure_lfp_onsets, input=stim_frame)[0]
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
    print(len(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys()))
    allopticalResults.save()

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

sys.exit()
