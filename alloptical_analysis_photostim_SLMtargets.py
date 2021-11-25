## script dedicated to code that focuses on analysis re: SLM targets data

# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys;
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


# 1) adding slm targets responses to alloptical results allopticalResults.slmtargets_stim_responses

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def add_slmtargets_responses_tracedFF(**kwargs):
    print("|- adding slm targets trace dFF responses to allopticalResults.slmtargets_stim_responses")
    print(kwargs)
    expobj = kwargs['expobj'] if 'expobj' in kwargs.keys() else KeyError('need to provide expobj as keyword argument')

    if 'pre' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean(
            [[np.mean(expobj.responses_SLMtargets_tracedFF[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

    elif 'post' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_outsz[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF_outsz.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean(
            [[np.mean(expobj.responses_SLMtargets_tracedFF_outsz[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

        if expobj.stims_in_sz > 1:
            prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"

            dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_insz[i]) for i in range(expobj.n_targets_total)]])
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets) insz'] = dFstdF_response

            reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_outsz.values()))
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF)) insz'] = reliability

            delta_trace_dFF_response = np.mean(
                [[np.mean(expobj.responses_SLMtargets_tracedFF_outsz[i]) for i in range(expobj.n_targets_total)]])
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets) insz'] = delta_trace_dFF_response


add_slmtargets_responses_tracedFF()

allopticalResults.slmtargets_stim_responses
# allopticalResults.save()


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

# %% 2.1) DATA COLLECTION SLMTargets: organize SLMTargets stim responses - across all appropriate run_pre4ap_trials, run_post4ap_trials trial comparisons - responses are dF/prestimF
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
    date = list(allopticalResults.metainfo.loc[
                    allopticalResults.metainfo['prep_trial'] == '%s %s' % (prep, pre4aptrial), 'date'])[0]
    print(f"\n{i}, {date}, {prep}")

    # skipping some trials that need fixing of the expobj
    if f"{prep} {pre4aptrial}" not in trials_skip:

        # load up pre-4ap trial
        print(f'|-- importing {prep} {pre4aptrial} - run_pre4ap_trials trial')

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
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                    prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'|-- importing {prep} {pre4aptrial_} - run_pre4ap_trials trial')
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
        print(f'|-- importing {prep} {post4aptrial} - run_post4ap_trials trial')
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
                        date = list(allopticalResults.metainfo.loc[
                                        allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                        prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'|-- importing {prep} {post4aptrial_} - run_post4ap_trials trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
                                                                   verbose=False)
                        if hasattr(expobj, 'responses_SLMtargets_outsz'):
                            df_ = expobj.responses_SLMtargets_outsz.T
                            # append additional dataframe to the first dataframe
                            df.append(df_, ignore_index=True)
                        else:
                            print('|-- **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (
                            post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\-- ***** skipping: {prep} {post4aptrial_}")

            stim_responses_comparisons_dict[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** 1 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (
            post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))

        ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells
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
                            date = list(allopticalResults.metainfo.loc[
                                            allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                            prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
                                                                       verbose=False)
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
                                    '**** 4 need to run collecting in sz responses SLMtargets attr for %s %s ****' % (
                                    post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} {post4aptrial_}")

                stim_responses_comparisons_dict[prep][f"{comparison_number}"]['in sz'] = df
            else:
                print('**** 4 need to run collecting insz responses SLMtargets attr for %s %s ****' % (
                post4aptrial, prep))
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
    stim_responses_comparisons_dict[prep][new_key] = stim_responses_comparisons_dict[prep].pop(f'{comparison_number}')
    # stim_responses_comparisons_dict[prep][new_key]= stim_responses_comparisons_dict[prep][f'{comparison_number}']

# save to: allopticalResults.stim_responses_comparisons
allopticalResults.stim_responses_comparisons = stim_responses_comparisons_dict
allopticalResults.save()

# %% 2.2) DATA COLLECTION SLMTargets: organize SLMTargets stim responses - across all appropriate run_pre4ap_trials, run_post4ap_trials trial comparisons - using whole trace dFF responses
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
    date = \
    allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == f"{prep} {pre4aptrial}", 'date'].values[
        0]
    print("\n\n\n-------------------------------------------")
    print(f"{i}, {date}, {prep}, run_pre4ap_trials trial: {pre4aptrial}, run_post4ap_trials trial: {post4aptrial}")

    # skipping some trials that need fixing of the expobj
    if f"{prep} {pre4aptrial}" not in trials_skip:

        # load up pre-4ap trial
        print(f'|-- importing {prep} {pre4aptrial} - run_pre4ap_trials trial')

        expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep, verbose=False,
                                                   do_processing=False)
        # collect raw Flu data from SLM targets
        expobj.collect_traces_from_targets(force_redo=False)
        aoutils.run_alloptical_processing_photostim(expobj, plots=False,
                                                    force_redo=False)  # REVIEW PROGRESS: run_pre4ap_trials seems to be working fine till here for trace_dFF processing

        df = expobj.responses_SLMtargets_tracedFF.T  # df == stim frame x cells (photostim targets)
        if len(allopticalResults.pre_4ap_trials[i]) > 1:
            for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
                print(f"\---- {i}, {j}")
                # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
                prep = allopticalResults.pre_4ap_trials[i][j][:-6]
                pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
                if f"{prep} {pre4aptrial}" not in trials_skip:
                    print(f"\------ adding trial to this comparison: {pre4aptrial_} [1.0]")
                    date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                    prep, pre4aptrial_), 'date'])[0]

                    # load up pre-4ap trial
                    print(f'\------ importing {prep} {pre4aptrial_} - run_pre4ap_trials trial')
                    expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False,
                                                               do_processing=False)
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
        stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}'] = {
            'pre-4ap': {}}  # initialize dict for saving responses
        stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['pre-4ap'] = df

        pre_4ap_df = df


    else:
        print(f"|-- skipping: {prep} run_pre4ap_trials trial {pre4aptrial}")

    ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS
    print(f'TEST 1.1 - working on {prep}, run_post4ap_trials trial {post4aptrial}')

    # skipping some trials that need fixing of the expobj
    if f"{prep} {post4aptrial}" not in trials_skip:

        # load up post-4ap trial and stim responses
        print(f'|-- importing {prep} {post4aptrial} - run_post4ap_trials trial')
        expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False,
                                                   do_processing=False)
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
                        date = list(allopticalResults.metainfo.loc[
                                        allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                        prep, pre4aptrial), 'date'])[0]

                        # load up post-4ap trial and stim responses
                        print(f'\------ importing {prep} {post4aptrial_} - run_post4ap_trials trial')
                        expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
                                                                   verbose=False, do_processing=False)
                        # collect raw Flu data from SLM targets
                        expobj.collect_traces_from_targets(force_redo=False)
                        aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)

                        if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
                            df_ = expobj.responses_SLMtargets_tracedFF_outsz.T
                            # append additional dataframe to the first dataframe
                            df.append(df_, ignore_index=True)
                        else:
                            print(
                                '\------ **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (
                                post4aptrial_, prep))
                            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
                    else:
                        print(f"\---- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial_}")

            stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['post-4ap'] = df

        else:
            print('\-- **** need to run collecting outsz responses SLMtargets attr for %s %s **** [1]' % (
            post4aptrial, prep))
            allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))

        ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells
        # post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:] -- same as run_post4ap_trials OUTSZ for loop one above

        # skipping some trials that need fixing of the expobj
        # if f"{prep} {post4aptrial}" not in trials_skip:
        #     print(f'TEST 1.2 - working on {prep} {post4aptrial}')

        # using the same skip statement as in the main for loop here

        # load up post-4ap trial and stim responses
        expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False,
                                                   do_processing=False)
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
                            date = list(allopticalResults.metainfo.loc[
                                            allopticalResults.metainfo['prep_trial'] == '%s %s' % (
                                            prep, pre4aptrial), 'date'])[0]

                            # load up post-4ap trial and stim responses
                            expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
                                                                       verbose=False, do_processing=False)
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
                                    '**** need to run collecting in sz responses SLMtargets attr for %s %s **** [4]' % (
                                    post4aptrial_, prep))
                                allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
                        else:
                            print(f"\-- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial_}")

                stim_responses_tracedFF_comparisons_dict[prep][f"{comparison_number}"]['in sz'] = df
            else:
                print('**** need to run collecting insz responses SLMtargets attr for %s %s **** [4]' % (
                post4aptrial, prep))
                allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
        else:
            print(f"**** need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial} [5]")

    else:
        print(f"\-- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial}")
        if not hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
            print(
                f'\-- **** need to run collecting outsz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [1]')

        if not hasattr(expobj, 'slmtargets_szboundary_stim'):
            print(
                f'**** need to run collecting insz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [2]')
        if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
            print(
                f'**** need to run collecting in sz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [3]')

    ## switch out the comparison_number to something more readable
    new_key = f"{pre4aptrial} vs. {post4aptrial}"
    stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep].pop(
        f'{comparison_number}')
    # stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']

    # save to: allopticalResults.stim_responses_tracedFF_comparisons
    allopticalResults.stim_responses_tracedFF_comparisons = stim_responses_tracedFF_comparisons_dict
    allopticalResults.save()

# %% 3.1) DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.post_4ap_trials + allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_out_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (outsz) [1.4.2] ***********')
                    # expobj.StimSuccessRate_SLMtargets_outsz, expobj.hits_SLMtargets_outsz, expobj.responses_SLMtargets_outsz, expobj.traces_SLMtargets_successes_outsz = \
                    #     expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stims_out_sz)
                    success_responses = expobj.hits_SLMtargets_outsz * expobj.responses_SLMtargets_outsz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[
                                                                        j], 'mean dFF response outsz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())

                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_in_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (insz) [1.4.3] ***********')
                    # expobj.StimSuccessRate_SLMtargets_insz, expobj.hits_SLMtargets_insz, expobj.responses_SLMtargets_insz, expobj.traces_SLMtargets_successes_insz = \
                    #     expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stims_in_sz)

                    success_responses = expobj.hits_SLMtargets_insz * expobj.responses_SLMtargets_insz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[
                                                                        j], 'mean dFF response insz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())


            elif 'pre' in expobj.metainfo['exptype']:
                seizure_filter = False
                print('\n Calculating stim success rates and response magnitudes [1.4.4] ***********')
                # expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
                #     expobj.get_SLMTarget_responses_dff(threshold=10, stims_to_use=expobj.stim_start_frames)

                success_responses = expobj.hits_SLMtargets * expobj.responses_SLMtargets
                success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[
                                                                    j], 'mean dFF response (hits, all targets)'] = success_responses.mean()
                print(success_responses.mean())

            expobj.save()
    allopticalResults.save()

# %% 3.2)  DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets_outsz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_outsz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[
                                                                    j], 'mean dFF response outsz (failures, all targets)'] = failure_responses.mean()

                inverse = (expobj.hits_SLMtargets_insz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_insz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[
                                                                    j], 'mean dFF response insz (failures, all targets)'] = failure_responses.mean()

            elif 'pre' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[
                                                                    j], 'mean dFF response (failures, all targets)'] = failure_responses.mean()
    allopticalResults.save()

# %% 5.0-main) collect SLM targets responses for stims dynamically over time

"""# plot the target photostim responses for individual targets for each stim over the course of the trial
#    (normalize to each target's overall mean response) and plot over the timecourse of the trial

# # ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
# for key in ls(allopticalResults.trial_maps['post'].keys())[-5:]:
#     for j in range(len(allopticalResults.trial_maps['post'][key])):
#         # import expobj
#         expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j))


# ls = ['RL108 t-013', 'RL109 t-021', 'RL109 t-016']
# # ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
# for key in ls(allopticalResults.trial_maps['post'].keys())[-5:]:
#     for j in range(len(allopticalResults.trial_maps['post'][key])):
#         # import expobj
#         expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j), do_processing=True)
"""

# %% 5.1) collect SLM targets responses for stims dynamically over time - APPROACH #1 - CALCULATING RESPONSE MAGNITUDE AT EACH STIM PER TARGET
key = 'e'
j = 0
exp = 'post'
expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.{j}")

SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
target_colors = pj.make_random_color_array(len(SLMtarget_ids))

# --- plot with mean FOV fluorescence signal
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
fig, axs[0] = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', figsize=[20, 3],
                                         show=False,
                                         fig=fig, ax=axs[0])
ax2 = axs[0].twinx()

## calculate and plot the response magnitude for each target at each stim;
#   where response magnitude is classified as response of each target at a particular stim relative to the mean response from the whole trial
for target in expobj.responses_SLMtargets.index:
    mean_response = np.mean(expobj.responses_SLMtargets.iloc[target, :])
    # print(mean_response)
    for i in expobj.responses_SLMtargets.columns:
        response = expobj.responses_SLMtargets.iloc[target, i] - mean_response
        rand = np.random.randint(-15, 25, 1)[
            0]  # * 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
        ax2.scatter(x=expobj.stim_start_frames[i] + rand, y=response, color=target_colors[target], alpha=0.70, s=15,
                    zorder=4)
        ax2.axhline(y=0)
        ax2.set_ylabel('Response mag. (relative to mean)')
# for i in expobj.stim_start_frames:
#     plt.axvline(i)
fig, axs[1] = aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='Time', show=False, fig=fig, ax=axs[1])
ax2.margins(x=0)
fig.suptitle(f"Photostim responses - {exp}-4ap {expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}")
fig.show()

# %% 5.2) collect SLM targets responses for stims dynamically over time - APPROACH #2 - USING Z-SCORED PHOTOSTIM RESPONSES

print(f"---------------------------------------------------------")
print(f"plotting zscored photostim responses over the whole trial")
print(f"---------------------------------------------------------")

### PRE 4AP
trials = list(allopticalResults.trial_maps['pre'].keys())
fig, axs = plt.subplots(nrows=len(trials) * 2, ncols=1, figsize=[20, 6 * len(trials)])
counter = 0
for expprep in list(allopticalResults.stim_responses_zscores.keys())[:3]:
    for trials_comparisons in allopticalResults.stim_responses_zscores[expprep]:
        pre4ap_trial = trials_comparisons[:5]
        post4ap_trial = trials_comparisons[-5:]

        # PRE 4AP STUFF
        if f"{expprep} {pre4ap_trial}" in pj.flattenOnce(allopticalResults.pre_4ap_trials):
            pre4ap_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['pre-4ap']

            print(f"working on expobj: {expprep} {pre4ap_trial}, counter @ {counter}")
            expobj, experiment = aoutils.import_expobj(prep=expprep, trial=pre4ap_trial)

            SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
            target_colors = pj.make_random_color_array(len(SLMtarget_ids))
            # --- plot with mean FOV fluorescence signal
            # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
            ax = axs[counter]
            fig, ax = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames', show=False,
                                                 fig=fig, ax=ax)
            ax2 = ax.twinx()
            ## retrieve the appropriate zscored database - run_pre4ap_trials stims
            targets = [x for x in list(pre4ap_df.columns) if type(x) == str and '_z' in x]
            for target in targets:
                for stim_idx in pre4ap_df.index[:-2]:
                    # if i == 'pre':
                    #     stim_idx = expobj.stim_start_frames.index(stim_idx)  # MINOR BUG: appears that for some reason the stim_idx of the allopticalResults.stim_responses_zscores for pre-4ap are actually the frames themselves
                    response = pre4ap_df.loc[stim_idx, target]
                    rand = np.random.randint(-15, 25, 1)[
                        0]  # * 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                    ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response,
                                color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

            ax2.axhline(y=0)
            ax2.set_ylabel('Response mag. (zscored to run_pre4ap_trials)')
            ax2.margins(x=0)

            ax3 = axs[counter + 1]
            ax3_2 = ax3.twinx()
            fig, ax3, ax3_2 = aoplot.plot_lfp_stims(expobj, x_axis='Time', show=False, fig=fig, ax=ax3, ax2=ax3_2)

            counter += 2
            print(f"|- finished on expobj: {expprep} {pre4ap_trial}, counter @ {counter}\n")

fig.suptitle(f"Photostim responses - pre-4ap", y=0.99)
save_path_full = f"{save_path_prefix}/pre4ap_indivtrial_zscore_responses.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()

### POST 4AP
trials_to_plot = pj.flattenOnce(allopticalResults.post_4ap_trials)
fig, axs = plt.subplots(nrows=len(trials_to_plot) * 2, ncols=1, figsize=[20, 6 * len(trials_to_plot)])
post4ap_trials_stimresponses_zscores = list(allopticalResults.stim_responses_zscores.keys())
counter = 0
for expprep in post4ap_trials_stimresponses_zscores:
    for trials_comparisons in allopticalResults.stim_responses_zscores[expprep]:
        if len(allopticalResults.stim_responses_zscores[expprep][
                   trials_comparisons].keys()) > 2:  ## make sure that there are keys containing data for post 4ap and in sz
            pre4ap_trial = trials_comparisons[:5]
            post4ap_trial = trials_comparisons[-5:]

            # POST 4AP STUFF
            if f"{expprep} {post4ap_trial}" in trials_to_plot:
                post4ap_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['post-4ap']

                insz_df = allopticalResults.stim_responses_zscores[expprep][trials_comparisons]['in sz']

                print(f"working on expobj: {expprep} {post4ap_trial}, counter @ {counter}")
                expobj, experiment = aoutils.import_expobj(prep=expprep, trial=post4ap_trial)

                SLMtarget_ids = list(range(len(expobj.SLMTargets_stims_dfstdF)))
                target_colors = pj.make_random_color_array(len(SLMtarget_ids))
                # --- plot with mean FOV fluorescence signal
                # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[20, 6])
                ax = axs[counter]
                fig, ax = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color='white', x_axis='frames',
                                                     show=False, fig=fig, ax=ax)
                ax.margins(x=0)

                ax2 = ax.twinx()
                ## retrieve the appropriate zscored database - run_post4ap_trials (outsz) stims
                targets = [x for x in list(post4ap_df.columns) if type(x) == str and '_z' in x]
                assert len(targets) == len(SLMtarget_ids), print(
                    'mismatch in SLMtargets_ids and targets run_post4ap_trials out sz')
                for target in targets:
                    for stim_idx in post4ap_df.index[:-2]:
                        response = post4ap_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[
                            0]  # * 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                        assert not np.isnan(response)
                        ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response,
                                    color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

                ## retrieve the appropriate zscored database - insz stims
                targets = [x for x in list(insz_df.columns) if type(x) == str]
                assert len(targets) == len(SLMtarget_ids), print('mismatch in SLMtargets_ids and targets in sz')
                for target in targets:
                    for stim_idx in insz_df.index:
                        response = insz_df.loc[stim_idx, target]
                        rand = np.random.randint(-15, 25, 1)[
                            0]  # * 1/(abs(response)**1/2)  # jittering around the stim_frame for the plot
                        if not np.isnan(response):
                            ax2.scatter(x=expobj.stim_start_frames[stim_idx] + rand, y=response,
                                        color=target_colors[targets.index(target)], alpha=0.70, s=15, zorder=4)

                ax2.axhline(y=0)
                ax2.set_ylabel('Response mag. (zscored to run_pre4ap_trials)')
                ax2.margins(x=0)

                ax3 = axs[counter + 1]
                ax3_2 = ax3.twinx()
                fig, ax3, ax3_2 = aoplot.plot_lfp_stims(expobj, x_axis='Time', show=False, fig=fig, ax=ax3, ax2=ax3_2)

                counter += 2
                print(f"|- finished on expobj: {expprep} {post4ap_trial}, counter @ {counter}\n")

fig.suptitle(f"Photostim responses - post-4ap", y=0.99)
save_path_full = f"{save_path_prefix}/post4ap_indivtrial_zscore_responses.png"
print(f'\nsaving figure to {save_path_full}')
fig.savefig(save_path_full)
fig.show()

# %% 6.1) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: dF/prestimF - for loop over all experiments to collect responses in terms of sz onset time

stim_relative_szonset_vs_avg_response_alltargets_atstim = {}

for prep in allopticalResults.stim_responses_comparisons.keys():
    # prep = 'PS07'

    for key in list(allopticalResults.stim_responses_comparisons[prep].keys()):
        # key = list(allopticalResults.stim_responses_comparisons[prep].keys())[0]
        # comp = 2
        if 'post-4ap' in allopticalResults.stim_responses_comparisons[prep][key]:
            post_4ap_df = allopticalResults.stim_responses_comparisons[prep][key]['post-4ap']
            if len(post_4ap_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, run_post4ap_trials trial: {post4aptrial}')
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

                stim_relative_szonset_vs_avg_response_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(
                    stims_relative_sz)
                stim_relative_szonset_vs_avg_response_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(
                    post_4ap_df_zscore_stim_relative_to_sz['avg'].tolist())

allopticalResults.stim_relative_szonset_vs_avg_response_alltargets_atstim = stim_relative_szonset_vs_avg_response_alltargets_atstim
allopticalResults.save()

# %% 6.2) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: delta(dFF) from whole trace - for loop over all experiments to collect responses in terms of sz onset time

stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim = {}

for prep in allopticalResults.stim_responses_tracedFF_comparisons.keys():
    # prep = 'PS07's

    for key in list(allopticalResults.stim_responses_tracedFF_comparisons[prep].keys()):
        # key = list(allopticalResults.stim_responses_tracedFF_comparisons[prep].keys())[0]
        # comp = 2
        if 'post-4ap' in allopticalResults.stim_responses_tracedFF_comparisons[prep][key]:
            post_4ap_df = allopticalResults.stim_responses_tracedFF_comparisons[prep][key]['post-4ap']
            if len(post_4ap_df) > 0:
                post4aptrial = key[-5:]
                print(f'working on.. {prep} {key}, run_post4ap_trials trial: {post4aptrial}')
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

                stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[f"{prep} {post4aptrial}"][0].append(
                    stims_relative_sz)
                stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim[f"{prep} {post4aptrial}"][1].append(
                    post_4ap_df_zscore_stim_relative_to_sz['avg'].tolist())

    allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim = stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim
    print(
        f"length of allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim dict: {len(allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim.keys())}")
    allopticalResults.save()

# %% 7.0-main-dc) TODO collect targets responses for stims vs. distance (starting with old code)- low priority right now

key = 'e'
j = 0
exp = 'post'
expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.{j}")

# plot response magnitude vs. distance
for i in range(len(expobj.stim_times)):
    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
    s2pcells = expobj.cells_sz_stim[expobj.stim_start_frames[i]]
    target_coord = expobj.target_coords_all[target]
    min_distance = 1000
    for j in range(len(s2pcells)):
        dist = pj.calc_distance_2points(target_coord, tuple(expobj.stat[j]['med']))  # distance in pixels
        if dist < min_distance:
            min_distance = dist

fig1, ax1 = plt.subplots(figsize=[5, 5])
responses = []
distance_to_sz = []
responses_ = []
distance_to_sz_ = []
for target in expobj.responses_SLMtargets.keys():
    mean_response = np.mean(expobj.responses_SLMtargets[target])
    target_coord = expobj.target_coords_all[target]
    # print(mean_response)

    # calculate response magnitude at each stim time for selected target
    for i in range(len(expobj.stim_times)):
        # the response magnitude of the current SLM target at the current stim time (relative to the mean of the responses of the target over this trial)
        response = expobj.responses_SLMtargets[target][
                       i] / mean_response  # changed to division by mean response instead of substracting
        min_distance = pj.calc_distance_2points((0, 0), (expobj.frame_x,
                                                         expobj.frame_y))  # maximum distance possible between two points within the FOV, used as the starting point when the sz has not invaded FOV yet

        if hasattr(expobj, 'cells_sz_stim') and expobj.stim_start_frames[i] in list(
                expobj.cells_sz_stim.keys()):  # calculate distance to sz only for stims where cell locations in or out of sz boundary are defined in the seizures
            if expobj.stim_start_frames[i] in expobj.stims_in_sz:
                # collect cells from this stim that are in sz
                s2pcells_sz = expobj.cells_sz_stim[expobj.stim_start_frames[i]]

                # classify the SLM target as in or out of sz, if out then continue with mesauring distance to seizure wavefront,
                # if in sz then assign negative value for distance to sz wavefront
                sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (
                expobj.analysis_save_path, expobj.metainfo['trial'], expobj.stim_start_frames[i])

                in_sz_bool = expobj._InOutSz(cell_med=[target_coord[1], target_coord[0]],
                                             sz_border_path=sz_border_path)

                if expobj.stim_start_frames[i] in expobj.not_flip_stims:
                    flip = False
                else:
                    flip = True
                    in_sz_bool = not in_sz_bool

                if in_sz_bool is True:
                    min_distance = -1

                else:
                    ## working on add feature for edgecolor of scatter plot based on calculated distance to seizure
                    ## -- thinking about doing this as comparing distances between all targets and all suite2p ROIs,
                    #     and the shortest distance that is found for each SLM target is that target's distance to seizure wavefront
                    # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
                    if len(s2pcells_sz) > 0:
                        for j in range(len(s2pcells_sz)):
                            s2p_idx = expobj.cell_id.index(s2pcells_sz[j])
                            dist = pj.calc_distance_2points(target_coord, tuple(
                                [expobj.stat[s2p_idx]['med'][1], expobj.stat[s2p_idx]['med'][0]]))  # distance in pixels
                            if dist < min_distance:
                                min_distance = dist

        if min_distance > 600:
            distance_to_sz_.append(min_distance + np.random.randint(-10, 10, 1)[0] - 165)
            responses_.append(response)
        elif min_distance > 0:
            distance_to_sz.append(min_distance)
            responses.append(response)

# calculate linear regression line
ax1.plot(range(int(min(distance_to_sz)), int(max(distance_to_sz))),
         np.poly1d(np.polyfit(distance_to_sz, responses, 1))(range(int(min(distance_to_sz)), int(max(distance_to_sz)))),
         color='black')

ax1.scatter(x=distance_to_sz, y=responses, color='cornflowerblue',
            alpha=0.5, s=16,
            zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.scatter(x=distance_to_sz_, y=responses_, color='firebrick',
            alpha=0.5, s=16,
            zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
ax1.set_xlabel('distance to seizure front (pixels)')
ax1.set_ylabel('response magnitude')
ax1.set_title('')
fig1.show()

# %% 8.0-main) avg responses in space around photostim targets - pre vs. run_post4ap_trials


# %% archive-1) adding slm targets responses to alloptical results allopticalResults.slmtargets_stim_responses


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

    # save to results object:
    allopticalResults.slmtargets_stim_responses.loc[
        counter, 'prep_trial'] = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
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
