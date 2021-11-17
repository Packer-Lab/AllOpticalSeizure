# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys; import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats, signal
import statsmodels.api
import statsmodels as sm

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

from skimage import draw

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/Nontargets_responses_2021-11-11'
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
# %% 6.1-dc) DATA COLLECTION - absolute stim responses vs. TIME to seizure onset - for loop over all experiments to collect responses in terms of sz onset time

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
                    closest_sz_onset = pj.findClosest(ls=expobj.seizure_lfp_onsets, input=stim_frame)[0]
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

sys.exit()

# %% 6) DATA COLLECTION: organize SLMTargets stim responses - across all appropriate pre4ap, post4ap trial comparisons
""" doing it in this way so that its easy to use in the response vs. stim times relative to seizure onset code (as this has already been coded up)"""

trials_skip = [
    'RL108 t-011',
    'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet..
]

allopticalResults.outsz_missing = []
allopticalResults.insz_missing = []
allopticalResults.stim_responses = {}
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

        # accounting for multiple pre/post photostim setup comparisons within each prep
        if prep not in allopticalResults.stim_responses.keys():
            allopticalResults.stim_responses[prep] = {}
            comparison_number = 1
        else:
                comparison_number = len(allopticalResults.stim_responses[prep]) + 1

        allopticalResults.stim_responses[prep][f'{comparison_number}'] = {'pre-4ap': {}}
        allopticalResults.stim_responses[prep][f'{comparison_number}']['pre-4ap'] = df

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

            allopticalResults.stim_responses[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** 1 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))



    ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells - zscore to the mean and std of the same SLM target calculated from the pre-4ap trial
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

                allopticalResults.stim_responses[prep][f"{comparison_number}"]['in sz'] = df
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
    allopticalResults.stim_responses[prep][new_key]= allopticalResults.stim_responses[prep].pop(f'{comparison_number}')
    # allopticalResults.stim_responses[prep][new_key]= allopticalResults.stim_responses[prep][f'{comparison_number}']


allopticalResults.save()



# sys.exit()
"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""


# %% 1) lists of trials to analyse for pre4ap and post4ap trials within experiments

allopticalResults.pre_4ap_trials = [
    ['RL108 t-009'],
    ['RL108 t-010'],
    ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],  # - pickle truncated .21/10/18 - analysis func jupyter run on .21/11/12
    ['RL109 t-014'],
    ['PS04 t-012',  # 'PS04 t-014',  - not sure what's wrong with PS04, but the photostim and Flu are falling out of sync .21/10/09
     'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    ['PS07 t-009'],
    # ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],  # matching post4ap trial cannot be analysed
    ['PS06 t-011'],
    # ['PS06 t-012'],  # matching post4ap trial cannot be analysed
    # ['PS11 t-007'],
    ['PS11 t-010'],
    # ['PS17 t-005'],
    # ['PS17 t-006', 'PS17 t-007'],
    # ['PS18 t-006']
]

allopticalResults.post_4ap_trials = [
    ['RL108 t-013'],
    ['RL108 t-011'],
    ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016', 'RL109 t-017'],
    ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    ['PS07 t-017'],
    # ['PS06 t-014', 'PS06 t-015'], - missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    ['PS06 t-013'],
    # ['PS06 t-016'], - no seizures, missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    # ['PS11 t-016'],
    ['PS11 t-011'],
    # ['PS17 t-011'],
    # ['PS17 t-009'],
    # ['PS18 t-008']
]

assert len(allopticalResults.pre_4ap_trials) == len(allopticalResults.post_4ap_trials), print('pre trials %s ' % len(allopticalResults.pre_4ap_trials),
                                                                                              'post trials %s ' % len(allopticalResults.post_4ap_trials))


allopticalResults.trial_maps = {'pre': {}, 'post': {}}
allopticalResults.trial_maps['pre'] = {
    'a': ['RL108 t-009'],
    # 'b': ['RL108 t-010'],
    # 'c': ['RL109 t-007'],
    'd': ['RL109 t-008'],
    'e': ['RL109 t-013'],
    'f': ['RL109 t-014'],
    # 'g': ['PS04 t-012',  # 'PS04 t-014',  # - temp just until PS04 gets reprocessed
    #       'PS04 t-017'],
    'h': ['PS05 t-010'],
    'i': ['PS07 t-007'],
    'j': ['PS07 t-009'],
    # 'k': ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    'l': ['PS06 t-011'],
    # 'm': ['PS06 t-012'],  # - t-016 missing sz lfp onsets
    # 'n': ['PS11 t-007'],
    'o': ['PS11 t-010'],
    # 'p': ['PS17 t-005'],
    # 'q': ['PS17 t-006', 'PS17 t-007'],
    # 'r': ['PS18 t-006']
}

allopticalResults.trial_maps['post'] = {
    'a': ['RL108 t-013'],
    # 'b': ['RL108 t-011'], -- need to redo sz boundary classifying processing
    # 'c': ['RL109 t-020'], -- need to redo sz boundary classifying processing
    'd': ['RL109 t-021'],
    'e': ['RL109 t-018'],
    'f': ['RL109 t-016', 'RL109 t-017'],
    # 'g': ['PS04 t-018'],  -- need to redo sz boundary classifying processing
    'h': ['PS05 t-012'],
    'i': ['PS07 t-011'],
    'j': ['PS07 t-017'],
    # 'k': ['PS06 t-014', 'PS06 t-015'],  # - missing seizure_lfp_onsets
    'l': ['PS06 t-013'],
    # 'm': ['PS06 t-016'],  # - missing seizure_lfp_onsets - LFP signal not clear, but there is seizures on avg Flu trace
    # 'n': ['PS11 t-016'],
    'o': ['PS11 t-011'],
    # 'p': ['PS17 t-011'],
    # 'q': ['PS17 t-009'],
    # 'r': ['PS18 t-008']
}

assert len(allopticalResults.trial_maps['pre'].keys()) == len(allopticalResults.trial_maps['post'].keys())

allopticalResults.save()





# %% 2) adding slm targets responses to alloptical results superobject.slmtargets_stim_responses

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




# %% 3) make a metainfo attribute to store all metainfo types of info for all experiments/trials
allopticalResults.metainfo = allopticalResults.slmtargets_stim_responses.loc[:, ['prep_trial', 'date', 'exptype']]



# %%#### -------------------- ALL OPTICAL PHOTOSTIM ANALYSIS ################################################

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
# %% 5.0-main)  RUN DATA ANALYSIS OF NON TARGETS:
# #  - Analysis of responses of non-targets from suite2p ROIs in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz boundary)


# # import expobj
# expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')
# aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', do_processing=False,
#                                          save_plot_suffix='Nontargets_responses_2021-10-24/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


# expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
# expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
#     aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim_sec=expobj.pre_stim_sec,
#                                             post_stim_sec=expobj.post_stim_sec)

# 5.0.1) re-calculating and plotting of excluded s2p ROIs and SLM target coordinates

for (i, key, j) in code_run_list_all:
    if (i, key, j) in short_list_pre or (i, key, j) in short_list_post:
        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')

        if not hasattr(expobj, 's2p_nontargets'):
            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(force_redo=True, plot=False)
            expobj.save()
        assert hasattr(expobj, 's2p_nontargets')
        save_path = save_path_prefix + f"/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - s2p ROIs plot.png"
        aoplot.s2pRoiImage(expobj, save_fig=save_path)



# %% 5.1) for loop to go through each expobj to analyze nontargets - pre4ap trials

# ls = ['PS05 t-010', 'PS06 t-011', 'PS11 t-010', 'PS17 t-005', 'PS17 t-006', 'PS17 t-007', 'PS18 t-006']
ls = pj.flattenOnce(allopticalResults.pre_4ap_trials)
for (i, key, j) in code_run_list_pre:
    # import expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, j))
    aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', do_processing=True, to_plot=True,
                                             save_plot_suffix=f"{save_path_prefix[-31:]}/{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-pre4ap.png")


# test: adding correct stim filters when analysing data to exclude stims/cells in seizure boundaries - this should be done, but not thouroughly tested necessarily yet //
# 5.1) for loop to go through each expobj to analyze nontargets - post4ap trials
# ls = ['RL108 t-013', 'RL109 t-021', 'RL109 t-016']
missing_slmtargets_sz_stim = []
ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
for (i, key, j) in code_run_list_all:
    # import expobj
    expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j), do_processing=True)
    if hasattr(expobj, 'slmtargets_szboundary_stim'):
        aoutils.run_allopticalAnalysisNontargetsPost4ap(expobj, normalize_to='pre-stim', do_processing=True, to_plot=True,
                                                        save_plot_suffix=f"{save_path_prefix[-31:]}/{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-post4ap.png")
    else:
        missing_slmtargets_sz_stim.append(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}")


# %% 5.2) collect average stats for each prep, and summarize into the appropriate data point
num_sig_responders = pd.DataFrame(columns=['pre4ap_pos', 'pre4ap_neg', 'post4ap_pos', 'post4ap_neg', '# of suite2p ROIs'])
possig_responders_traces = []
negsig_responders_traces = []

allopticalResults.pre_stim_sec = 0.5
allopticalResults.stim_dur_sec = 0.25
allopticalResults.post_stim_sec = 3

for key in list(allopticalResults.trial_maps['pre'].keys()):
    name = allopticalResults.trial_maps['pre'][key][0][:-6]

    ########
    # pre-4ap trials calculations
    n_trials = len(allopticalResults.trial_maps['pre'][key])
    pre4ap_possig_responders_avgresponse = []
    pre4ap_negsig_responders_avgresponse = []
    pre4ap_num_pos = 0
    pre4ap_num_neg = 0
    for i in range(n_trials):
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, i))
        if sum(expobj.sig_units) > 0:
            name += expobj.metainfo['trial']
            pre4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            pre4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

            if i == 0:  # only taking trials averages from the first sub-trial from the matched comparisons
                pre4ap_possig_responders_avgresponse.append(pre4ap_possig_responders_avgresponse_)
                pre4ap_negsig_responders_avgresponse.append(pre4ap_negsig_responders_avgresponse_)
                stim_dur_fr = int(np.ceil(allopticalResults.stim_dur_sec * expobj.fps))  # setting 500ms as the dummy standardized stimduration
                pre_stim_fr = int(np.ceil(allopticalResults.pre_stim_sec * expobj.fps))  # setting the pre_stim array collection period again hard
                post_stim_fr = int(np.ceil(allopticalResults.post_stim_sec * expobj.fps))  # setting the post_stim array collection period again hard

                data_traces = []
                for trace in pre4ap_possig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[0] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                pre4ap_possig_responders_avgresponse = np.array(data_traces); print(f"shape of pre4ap_possig_responders array {pre4ap_possig_responders_avgresponse.shape} [5.5-1]")
                # print('stop here... [5.5-1]')

                data_traces = []
                for trace in pre4ap_negsig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[0] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                pre4ap_negsig_responders_avgresponse = np.array(data_traces); print(f"shape of pre4ap_negsig_responders array {pre4ap_negsig_responders_avgresponse.shape} [5.5-2]")
                # print('stop here... [5.5-2]')


                pre4ap_num_pos += len(pre4ap_possig_responders_avgresponse_)
                pre4ap_num_neg += len(pre4ap_negsig_responders_avgresponse_)
        else:
            pre4ap_num_pos += 0
            pre4ap_num_neg += 0
    pre4ap_num_pos = pre4ap_num_pos / n_trials
    pre4ap_num_neg = pre4ap_num_neg / n_trials


    num_suite2p_rois = len(expobj.good_cells)  # this will be the same number for pre and post4ap (as they should be the same cells)


    # post-4ap trials calculations
    name += 'vs.'
    n_trials = len(allopticalResults.trial_maps['post'][key])
    post4ap_possig_responders_avgresponse = []
    post4ap_negsig_responders_avgresponse = []
    post4ap_num_pos = 0
    post4ap_num_neg = 0
    for i in range(n_trials):
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, i))
        if sum(expobj.sig_units) > 0:
            name += expobj.metainfo['trial']
            post4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            post4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

            if i == 0:
                post4ap_possig_responders_avgresponse.append(post4ap_possig_responders_avgresponse_)
                post4ap_negsig_responders_avgresponse.append(post4ap_negsig_responders_avgresponse_)
                stim_dur_fr = int(np.ceil(allopticalResults.stim_dur_sec * expobj.fps))  # setting 500ms as the dummy standardized stimduration
                pre_stim_fr = int(np.ceil(allopticalResults.pre_stim_sec * expobj.fps))  # setting the pre_stim array collection period again hard
                post_stim_fr = int(np.ceil(allopticalResults.post_stim_sec * expobj.fps))  # setting the post_stim array collection period again hard


                data_traces = []
                for trace in post4ap_possig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[0] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                post4ap_possig_responders_avgresponse = np.array(data_traces); print(f"shape of post4ap_possig_responders array {post4ap_possig_responders_avgresponse.shape} [5.5-3]")
                # print('stop here... [5.5-3]')


                data_traces = []
                for trace in post4ap_negsig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[0] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                post4ap_negsig_responders_avgresponse = np.array(data_traces); print(f"shape of post4ap_negsig_responders array {post4ap_negsig_responders_avgresponse.shape} [5.5-4]")
                # print('stop here... [5.5-4]')


                post4ap_num_pos += len(post4ap_possig_responders_avgresponse_)
                post4ap_num_neg += len(post4ap_negsig_responders_avgresponse_)
        else:
            post4ap_num_pos += 0
            post4ap_num_neg += 0
    post4ap_num_pos = post4ap_num_pos / n_trials
    post4ap_num_neg = post4ap_num_neg / n_trials


    ########
    # place into the appropriate dataframe
    series = [pre4ap_num_pos, pre4ap_num_neg, post4ap_num_pos, post4ap_num_neg, int(num_suite2p_rois)]
    num_sig_responders.loc[name] = series

    # place sig pos and sig neg traces into the ls
    possig_responders_traces.append([pre4ap_possig_responders_avgresponse, post4ap_possig_responders_avgresponse])
    negsig_responders_traces.append([pre4ap_negsig_responders_avgresponse, post4ap_negsig_responders_avgresponse])

allopticalResults.num_sig_responders_df = num_sig_responders
allopticalResults.possig_responders_traces = np.asarray(possig_responders_traces)
allopticalResults.negsig_responders_traces = np.asarray(negsig_responders_traces)

allopticalResults.save()







#########################################################################################################################
#### END OF CODE THAT HAS BEEN REVIEWED SO FAR ##########################################################################
#########################################################################################################################

#%%


















































# %% define cells in proximity of the targeted cell and plot the flu of those pre and post-4ap
# - maybe make like a heatmap around the cell that is being photostimed
# Action plan:
# - make a dictionary for every cell that was targeted (and found in suite2p) that contains:
#   - coordinates of the cell
#   - trials that were successful in raising the fluorescence at least 30% over pre-stim period
#   - other cells that are in 300um proximity of the targeted cell

# same as calculating repsonses and assigning to pixel areas, but by coordinates now
group = 0
responses_group_1_ = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')
for n in filter(lambda n: n not in expobj.good_photostim_cells_all, expobj.good_cells):
    idx = expobj.cell_id.index(n)
    ypix = int(expobj.stat[idx]['med'][0])
    xpix = int(expobj.stat[idx]['med'][1])
    responses_group_1_[ypix, xpix] = 100 + 1 * round(average_responses[group][expobj.good_cells.index(n)], 2)

pixels_200 = round(200. / expobj.pix_sz_x)
pixels_20 = round(20. / expobj.pix_sz_x)

prox_responses = np.zeros((pixels_200 * 2, pixels_200 * 2), dtype='uint16')
for cell in expobj.good_photostim_cells_all:
    # cell = expobj.good_photostim_cells_all[0]
    # define annulus around the targeted cell
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    rr, cc = draw.circle(y, x, radius=pixels_200, shape=arr.shape)
    arr[rr, cc] = 1
    rr, cc = draw.circle(y, x, radius=pixels_20, shape=arr.shape)
    arr[rr, cc] = 0
    # plt.imshow(arr); plt.show() # check shape of the annulus

    # find all cells that are not photostim targeted cells, and are in proximity to the cell of interest
    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            loc_ = [pixels_200 + y_loc - y, pixels_200 + x_loc - x]
            prox_responses[loc_[0] - 2:loc_[0] + 2, loc_[1] - 2:loc_[1] + 2] = responses_group_1_[y_loc, x_loc]
            # prox_responses[loc_[0], loc_[1]] = responses_group_1_[y_loc, x_loc]
        prox_responses[pixels_200 - pixels_20:pixels_200 + pixels_20,
        pixels_200 - pixels_20:pixels_200 + pixels_20] = 500  # add in the 20um box around the cell of interest

prox_responses = np.ma.masked_where(prox_responses < 0.05, prox_responses)
cmap = plt.cm.bwr
cmap.set_bad(color='black')

plt.imshow(prox_responses, cmap=cmap)
cb = plt.colorbar()
cb.set_label('dF/preF')
plt.clim(80, 120)
plt.suptitle((experiment + '- avg. stim responses - Group %s' % group), y=1.00)
plt.show()



# %%
# plot response over distance from photostim. target cell to non-target cell in proximity
import math

d = {}
d['cell_pairs'] = []
d['distance'] = []
d['response_of_target'] = []
d['response_of_non_target'] = []
for cell in expobj.good_photostim_cells[0]:
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    rr, cc = draw.circle(y, x, radius=pixels_200, shape=arr.shape)
    arr[rr, cc] = 1
    rr, cc = draw.circle(y, x, radius=pixels_20, shape=arr.shape)
    arr[rr, cc] = 0  # delete selecting from the 20um around the targeted cell

    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            d['cell_pairs'].append('%s_%s' % (cell, cell2))
            d['distance'].append(math.hypot(y_loc - y, x_loc - x) * expobj.pix_sz_x)
            d['response_of_target'].append(average_responses[0][expobj.good_cells.index(cell)])
            d['response_of_non_target'].append(average_responses[0][expobj.good_cells.index(cell2)])

df_dist_resp = pd.DataFrame(d)

# plot distance vs. photostimulation response
plt.figure()
plt.scatter(x=df_dist_resp['distance'], y=df_dist_resp['response_of_non_target'])
plt.show()


# %%
# TODO calculate probability of stimulation in 100x100um micron bins around targeted cell

all_x = []
all_y = []
for cell2 in expobj.good_cells:
    y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
    x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
    all_x.append(x_loc)
    all_y.append(y_loc)


def binned_amplitudes_2d(all_x, all_y, responses_of_cells, response_metric='dF/preF', bins=35, title=experiment):
    """
    :param all_x: ls of x coords of cells in dataset
    :param all_y: ls of y coords of cells in dataset
    :param responses_of_cells: ls of responses of cells to plots
    :param bins: integer - number of bins to split FOV in (along one axis)
    :return: plot of binned 2d histograms
    """

    all_amps_real = responses_of_cells  # ls of photostim. responses
    denominator, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins)
    numerator, _, _ = np.histogram2d(all_x, all_y, bins=bins, weights=all_amps_real)
    h = numerator / denominator  # divide the overall
    Y, X = np.meshgrid(xedges, yedges)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    mesh1 = ax[0].pcolormesh(X, Y, h, cmap='RdBu_r', vmin=-20.0, vmax=20.0)
    ax[0].set_xlabel('Cortical distance (um)')
    ax[0].set_ylabel('Cortical distance (um)')
    ax[0].set_aspect('equal')

    range_ = max(all_x) - min(all_x)  # range of pixel values
    labels = [item for item in ax[0].get_xticks()]
    length = expobj.pix_sz_x * range_
    for item in labels:
        labels[labels.index(item)] = int(round(item / range_ * length))
    ax[0].set_yticklabels(labels)
    ax[0].set_xticklabels(labels)

    # ax[0].set_ylim([expobj.pix_sz_x*expobj.frame_x, 0])
    ax[0].set_title('Binned responses (%s um bins)' % round(length / bins))
    cb = plt.colorbar(mesh1, ax=ax[0])
    cb.set_label(response_metric)

    mesh2 = ax[1].pcolormesh(X, Y, denominator.astype(int), cmap='inferno', vmin=0, vmax=np.amax(denominator))
    ax[1].set_xlabel('Cortical distance (um)')
    ax[1].set_ylabel('Cortical distance (um)')
    ax[1].set_aspect('equal')
    labels = [item for item in ax[1].get_xticks()]
    for item in labels:
        length = expobj.pix_sz_x * range_
        labels[labels.index(item)] = int(round(item / range_ * length))
    ax[1].set_yticklabels(labels)
    ax[1].set_xticklabels(labels)

    # ax[1].set_ylim([expobj.pix_sz_x*expobj.frame_x, 0])
    ax[1].set_title('Number of cells in bin; %s total cells' % len(responses_of_cells))
    cb = plt.colorbar(mesh2, ax=ax[1])
    cb.set_label('num cells')

    plt.suptitle(title, horizontalalignment='center', verticalalignment='top', y=1.0)
    plt.show()


binned_amplitudes_2d(all_x, all_y, responses_of_cells=average_responses[0],
                     title='%s - slm group1 - whole FOV' % experiment)  # 2d spatial binned responses of all cells in average_responses argument
binned_amplitudes_2d(all_x, all_y, responses_of_cells=average_responses[1],
                     title='%s - slm group2 - whole FOV' % experiment)  # 2d spatial binned responses of all cells in average_responses argument

group = 1
e = {}
e['cell_pairs'] = []
e['distance'] = []
e['response_of_target'] = []
e['response_of_non_target'] = []
e['norm_location - x'] = []
e['norm_location - y'] = []
for cell in expobj.good_photostim_cells[0]:
    # cell = expobj.good_photostim_cells[0][0]
    y = int(expobj.stat[expobj.cell_id.index(cell)]['med'][0])
    x = int(expobj.stat[expobj.cell_id.index(cell)]['med'][1])

    # make a square array around the cell of interest
    arr = np.zeros((expobj.frame_x, expobj.frame_x))
    coords = draw.rectangle(start=(y - pixels_200, x - pixels_200), extent=pixels_200 * 2)
    # coords = draw.rectangle(start=(0,100), extent=pixels_200)
    arr[coords] = 1
    coords = draw.rectangle(start=(y - pixels_20, x - pixels_20), extent=pixels_20 * 2)
    arr[coords] = 0
    # plt.imshow(arr); plt.show() # show the created array if needed

    for cell2 in filter(lambda cell2: cell2 not in expobj.good_photostim_cells_all, expobj.good_cells):
        y_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][0])
        x_loc = int(expobj.stat[expobj.cell_id.index(cell2)]['med'][1])
        if arr[y_loc, x_loc] == 1.0:
            e['norm_location - y'].append(round(pixels_200 + y_loc - y))
            e['norm_location - x'].append(round(pixels_200 + x_loc - x))
            e['cell_pairs'].append('%s_%s' % (cell, cell2))
            e['distance'].append(math.hypot(y_loc - y, x_loc - x) * expobj.pix_sz_x)
            e['response_of_target'].append(average_responses[group][expobj.good_cells.index(cell)])
            e['response_of_non_target'].append(average_responses[group][expobj.good_cells.index(
                cell2)])  # note that SLM group #1 has been hardcorded in! # #

df_dist_resp_rec = pd.DataFrame(e)

binned_amplitudes_2d(all_x=list(df_dist_resp_rec['norm_location - x']),
                     all_y=list(df_dist_resp_rec['norm_location - y']),
                     responses_of_cells=list(df_dist_resp_rec['response_of_non_target']), bins=20,
                     response_metric='dF/preF',
                     title=(
                             experiment + ' - slm group %s - targeted cell proximity' % group))  # 2d spatial binned repsonses of all cells in average_responses argument

# %%

# next multiply the annulus array with a matrix of cell coords (with responses) responses_group_1


# photostimulation of targeted cells before CSD, just after CSD, and a while after CSD


# photostimulation of targeted cells before seizure, just after seizure, and a while after seizure


# %%

cells_dff_exc = []
cells_dff_inh = []
for cell in expobj.good_cells:
    if cell in expobj.cell_id:
        cell_idx = expobj.cell_id.index(cell)
        flu = []
        for stim in stim_timings:
            # frames_to_plot = ls(range(stim-8, stim+35))
            flu.append(expobj.raw[cell_idx][stim - pre_stim:stim + post_stim])

        flu_dff = []
        for trace in flu:
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / mean) * 100
            flu_dff.append(trace_dff)

        all_cells_dff.append(np.mean(flu_dff, axis=0))

        thresh = np.mean(np.mean(flu_dff, axis=0)[pre_stim + 10:pre_stim + 100])
        if thresh > 30:
            good_std_cells.append(cell)
            good_std_cells_dff_exc.append(np.mean(flu_dff, axis=0))
        elif thresh < -30:
            good_std_cells.append(cell)
            good_std_cells_dff_inh.append(np.mean(flu_dff, axis=0))

        flu_std = []
        std = np.std(flu)
        mean = np.mean(flu[0:pre_stim])
        for trace in flu:
            df_stdf = (trace - mean) / std
            flu_std.append(df_stdf)

        # thresh = np.mean(np.mean(flu_std, axis=0)[pre_stim_sec+10:pre_stim_sec+30])
        #
        # if thresh > 1*std:
        #     good_std_cells.append(cell)
        #     good_std_cells_dff_exc.append(np.mean(flu_dff, axis=0))
        # elif thresh < -1*std:
        #     good_std_cells.append(cell)
        #     good_std_cells_dff_inh.append(np.mean(flu_dff, axis=0))

        print('Pre-stim mean:', mean)
        print('Pre-stim std:', std)
        print('Post-stim dff:', thresh)
        print('                            ')

        # flu_avg = np.mean(flu_dff, axis=0)
        # std = np.std(flu_dff, axis=0)
        # ci = 1.960 * (std/np.sqrt(len(flu_dff))) # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
        # x = ls(range(-pre_stim_sec, post_stim_sec))
        # y = flu_avg
        #
        # fig, ax = plt.subplots()
        # ax.fill_between(x, (y - ci), (y + ci), edgecolor='b', alpha=.1) # plot confidence interval
        # ax.axvspan(0, 10, alpha=0.2, edgecolor='red')
        # ax.plot(x, y)
        # fig.suptitle('Cell %s' % cell)
        # plt.show()

aoutils.plot_photostim_avg(dff_array=all_cells_dff, pre_stim=pre_stim, post_stim=post_stim, title=title)

################
cell_idx = expobj.cell_id.index(3863)
std = np.std(expobj.raw[cell_idx])
mean = np.mean(expobj.raw[cell_idx])

plt.figure(figsize=(50, 3))
fig, ax = plt.subplots()
ax.axhline(mean + 2.5 * std)
plt.plot(expobj.raw[cell_idx])
fig.show()


################# - ARCHIVED NOV 11 2021
# %% 4) ###### IMPORT pkl file containing data in form of expobj, and run processing as needed (implemented as a loop currently)


expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')

plot = 1
if plot:
    aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
    aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='time', figsize=[8, 2])
    aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered)
    aoplot.plot_lfp_stims(expobj)

for exptype in ['post', 'pre']:
    for key in allopticalResults.trial_maps[exptype].keys():
        if len(allopticalResults.trial_maps[exptype][key]) > 1:
            aoresults_map_id = []
            for i in range(len(allopticalResults.trial_maps[exptype][key])):
                aoresults_map_id.append('%s %s.%s' % (exptype, key, i))
        else:
            aoresults_map_id = ['%s %s' % (exptype, key)]

        for mapid in aoresults_map_id:
            expobj, experiment = aoutils.import_expobj(aoresults_map_id=mapid)

            plot = 0
            if plot:
                aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
                aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='time', figsize=[8, 2])
                aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered)
                aoplot.plot_lfp_stims(expobj)

        # 5) any data processing -- if needed

        # expobj.paqProcessing()
        # expobj._findTargets()


        # if not hasattr(expobj, 's2p_path'):
        #     expobj.s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-1x-alltrials/plane0'

        # expobj.s2pProcessing(s2p_path=expobj.s2p_path, subset_frames=expobj.curr_trial_frames, subtract_neuropil=True,
        #                      baseline_frames=expobj.baseline_frames, force_redo=True)
            expobj._findTargetedS2pROIs(force_redo=True)
        # aoutils.s2pMaskStack(obj=expobj, pkl_list=[expobj.pkl_path], s2p_path=expobj.s2p_path, parent_folder=expobj.analysis_save_path, force_redo=True)
        #



            if not hasattr(expobj, 'meanRawFluTrace'):
                expobj.mean_raw_flu_trace(plot=True)

            # for suite2p detected non-ROIs
            expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
                expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
                aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                                        post_stim=expobj.post_stim)
            # for s2p detected target ROIs
            expobj.targets_dff, expobj.targets_dff_avg, expobj.targets_dfstdF, \
            expobj.targets_dfstdF_avg, expobj.targets_stims_raw, expobj.targets_stims_raw_avg = \
                aoutils.get_s2ptargets_stim_traces(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                                   post_stim=expobj.post_stim)


            expobj.save()




# %% suite2p ROIs - PHOTOSTIM TARGETS - PLOT AVG PHOTOSTIM PRE- POST- STIM TRACE AVGed OVER ALL PHOTOSTIM. TRIALS

to_plot = 'dFstdF'

if to_plot == 'dFstdF':
    arr = np.asarray([i for i in expobj.targets_dfstdF_avg])
    y_label = 'dFstdF (normalized to prestim period)'
elif to_plot == 'dFF':
    arr = np.asarray([i for i in expobj.targets_dff_avg])
    y_label = 'dFF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=arr, expobj=expobj, pre_stim_sec=0.5, post_stim_sec=1.0,
                              title=(experiment + '- responses of all photostim targets'), figsize=[5,4],
                              x_label='Time post-stimulation (seconds)')


# %% SUITE2P ROIS - PHOTOSTIM TARGETS - PLOT ENTIRE TRIAL - individual ROIs plotted individually entire Flu trace

to_plot = expobj.dff_SLMTargets
aoplot.plot_photostim_traces_overlap(array=to_plot, expobj=expobj, y_lims=[0, 5000], title=(experiment + '-'))

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames', y_label='dFF Flu',
                             title='%s %s - dFF SLM Targets' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


# # plot the photostim targeted cells as a heatmap
# dff_array = expobj.SLMTargets_dff[:, :]
# w = 10
# dff_array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in dff_array]
# dff_array = np.asarray(dff_array)
#
# plt.figure(figsize=(5, 10));
# sns.heatmap(dff_array, cmap='RdBu_r', vmin=0, vmax=500);
# plt.show()



# %% SLM PHOTOSTIM TARGETS - plot individual, full traces, dff normalized

# make rolling average for these plots to smooth out the traces a little more
w = 3
to_plot = np.asarray([(np.convolve(trace, np.ones(w), 'valid') / w) for trace in expobj.dff_SLMTargets])
# to_plot = expobj.dff_SLMTargets

aoplot.plot_photostim_traces(array=to_plot, expobj=expobj, x_label='Frames',
                             y_label='dFF Flu', title=experiment)

aoplot.plot_photostim_traces_overlap(array=expobj.dff_SLMTargets, expobj=expobj, x_axis='Time (secs.)',
                                     title='%s - dFF Flu photostims' % experiment, figsize=(2*20, 2*len(to_plot)*0.15))

# len_ = len(array)
# fig, axs = plt.subplots(nrows=len_, sharex=True, figsize=(30, 3 * len_))
# for i in range(len(axs)):
#     axs[i].plot(array[i], linewidth=1, edgecolor='black')
#     for j in expobj.stim_start_frames:
#         axs[i].axvline(x=j, c='gray', alpha=0.7, linestyle='--')
#     if len_ == len(expobj.s2p_cell_targets):
#         axs[i].set_title('Cell # %s' % expobj.s2p_cell_targets[i])
# plt.show()

# array = (np.convolve(SLMTargets_stims_raw[targets_idx], np.ones(w), 'valid') / w)

# # targets_idx = 0
# plot = True
# for i in range(0, expobj.n_targets_total):
#     SLMTargets_stims_raw, SLMTargets_stims_dff, SLMtargets_stims_dfstdF = expobj.get_alltargets_stim_traces_norm(targets_idx=i, pre_stim_sec=pre_stim_sec,
#                                                                                                                  post_stim_sec=post_stim_sec)
#     if plot:
#         w = 2
#         array = [(np.convolve(trace, np.ones(w), 'valid') / w) for trace in SLMTargets_stims_raw]
#         random_sub = np.random.randint(0,100,10)
#         aoplot.plot_periphotostim_avg(arr=SLMtargets_stims_dfstdF[random_sub], expobj=expobj, stim_duration=expobj.stim_duration_frames,
#                                       title='Target ' + str(i), pre_stim_sec=pre_stim_sec, post_stim_sec=post_stim_sec, color='steelblue', y_lims=[-0.5, 2.5])
#     # plt.show()


# x = np.asarray([i for i in expobj.good_photostim_cells_stim_responses_dFF[0]])
# x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])

y_label = 'dF/prestim_stdF'
aoplot.plot_periphotostim_avg(arr=expobj.SLMTargets_stims_dfstdF_avg, expobj=expobj, stim_duration=expobj.stim_duration_frames,
                              figsize=[5, 4], y_lims=[-0.5, 3], pre_stim_sec=0.5, post_stim_sec=1.0,
                              title=(experiment + '- responses of all photostim targets'),
                              y_label=y_label, x_label='post-stimulation (seconds)')


# %% plotting of photostim. success rate

data = [np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]

pj.plot_hist_density([data], x_label='response magnitude (dF/stdF)')
pj.plot_bar_with_points(data=[list(expobj.StimSuccessRate_SLMtargets.values())], x_tick_labels=[expobj.metainfo['trial']], ylims=[0, 100], bar=False, y_label='% success stims.',
                        title='%s success rate of stim responses' % expobj.metainfo['trial'], expand_size_x=2)



# %% SUITE2P NON-TARGETS - PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
x = np.asarray([i for i in expobj.dfstdF_traces_avg])
# y_label = 'pct. dFF (normalized to prestim period)'
y_label = 'dFstdF (normalized to prestim period)'

aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=0.5,
                              post_stim_sec=1.5, title='responses of s2p non-targets', y_label=y_label,
                              x_label='Time post-stimulation (seconds)', y_lims=[-1, 3])


# %% PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D

arr = np.asarray([i for i in expobj.targets_dff_avg]); vmin = -1; vmax = 1
arr = np.asarray([i for i in expobj.targets_dff_avg]); vmin = -20; vmax = 20
aoplot.plot_traces_heatmap(arr, expobj=expobj, vmin=-20, vmax=20, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=('peristim avg trace heatmap' + ' - slm targets only'), x_label='Time')

arr = np.asarray([i for i in expobj.dfstdF_traces_avg]); vmin = -1; vmax = 1
arr = np.asarray([i for i in expobj.dff_traces_avg]); vmin = -20; vmax = 20
aoplot.plot_traces_heatmap(arr, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames - 1,
                           title=('peristim avg trace heatmap' + ' - nontargets'), x_label='Time')


# %% BAR PLOT PHOTOSTIM RESPONSES SIZE - TARGETS vs. NON-TARGETS
# collect photostim timed average dff traces
all_cells_dff = []
good_std_cells = []

# calculate and plot average response of cells in response to all stims as a bar graph


# there's a bunch of very high dFF responses of cells
# remove cells with very high average response values from the dff dataframe
# high_responders = expobj.average_responses_df[expobj.average_responses_df['Avg. dFF response'] > 500].index.values
# expobj.dff_responses_all_cells.iloc[high_responders[0], 1:]
# ls(expobj.dff_responses_all_cells.iloc[high_responders[0], 1:])
# idx = expobj.cell_id.index(1668);
# aoplot.plot_flu_trace(expobj=expobj, idx=idx, to_plot='dff', size_factor=2)


# need to troubleshoot how these scripts are calculating the post stim responses for the non-targets because some of them seem ridiculously off
# --->  this should be okay now since I've moved to df_stdf correct?

group1 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'photostim target'][
                  'Avg. dF/stdF response'])
group2 = list(
    expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'non-target']['Avg. dF/stdF response'])
pj.plot_bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6],
                        ylims=[0, 3], bar=False, colors=['red', 'black'], title=experiment, y_label='Avg dF/stdF response',
                        expand_size_y=2, expand_size_x=1)


# %% PLOT imshow() XY area locations with COLORS AS average response of ALL cells in FOV

aoplot.xyloc_responses(expobj, to_plot='dfstdf', clim=[-1, +1], plot_target_coords=True)


# %% PLOT INDIVIDUAL WHOLE TRACES AS HEATMAP OF PHOTOSTIM. RESPONSES TO PHOTOSTIM FOR ALL CELLS -- this is just the whole trace for each target, not avg over stims in any way
# - need to find a way to sort these responses that similar cells are sorted together
# - implement a heirarchical clustering method

stim_timings = [str(i) for i in expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

# make heatmap of responses across all cells across all stims
df_ = expobj.dfstdf_all_cells[stim_timings]  # select appropriate stim time reponses from the pandas df
df_ = df_[df_.columns].astype(float)

plt.figure(figsize=(5, 15));
sns.heatmap(df_, cmap='seismic', vmin=-5, vmax=5, cbar_kws={"shrink": 0.25});
plt.show()