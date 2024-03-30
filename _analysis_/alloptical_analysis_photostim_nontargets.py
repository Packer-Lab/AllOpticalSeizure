# script dedicated to code that focuses on analysis re: Non targets data

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys;

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd

from archive import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting as aoplot

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

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

# %% 0)
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


# %% 1.0-main) RUN DATA ANALYSIS OF NON TARGETS:
# #  - Analysis of responses of non-targets from suite2p ROIs in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz boundary)


# # import expobj
# expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')
# aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', do_processing=False,
#                                          save_plot_suffix='Nontargets_responses_2021-10-24/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


# expobj.dff_traces_nontargets, expobj.dff_traces_nontargets_avg, expobj.dfstdF_traces_nontargets, \
# expobj.dfstdF_traces_nontargets_avg, expobj.raw_traces_nontargets, expobj.raw_traces_nontargets_avg = \
#     aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim_sec=expobj.pre_stim_sec,
#                                             post_stim_sec=expobj.post_stim_sec)

# 1.0.1) re-calculating and plotting of excluded s2p ROIs and SLM target coordinates

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
        # save_path = save_path_prefix + f"/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - s2p ROIs plot.png"
        aoplot.s2pRoiImage(expobj, fig_save_name=f"/{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']} - s2p ROIs plot.png")



# %% 1.1) for loop to go through each expobj to analyze nontargets - run_pre4ap_trials trials - TODO - update using trace_dFF processed data

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False)
def run_allopticalNontargets(**kwargs):
    expobj = kwargs['expobj']
    if not hasattr(expobj, 's2p_nontargets'):
        expobj._parseNAPARMgpl()
        expobj._findTargetsAreas()
        expobj._findTargetedS2pROIs(force_redo=True, plot=False)
        expobj.save()

    aoutils.run_allopticalAnalysisNontargets(expobj=expobj, normalize_to='pre-stim', do_processing=True, to_plot=True,
                                             save_plot_suffix=f"{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-pre4ap.png")
run_allopticalNontargets()

# test: adding correct stim filters when analysing data to exclude stims/cells in seizure boundaries - this should be done, but not thouroughly tested necessarily yet //
# 2.1) for loop to go through each expobj to analyze nontargets - run_post4ap_trials trials
# ######### decorated ################################
# # ls = ['RL108 t-013', 'RL109 t-021', 'RL109 t-016']
# missing_slmtargets_sz_stim = []
# ls = pj.flattenOnce(allopticalResults.post_4ap_trials)
# for (i, key, j) in code_run_list_all:
#     # import expobj
#     expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j), do_processing=True)
# ######### decorated ################################

missing_slmtargets_sz_stim = []
@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, run_trials=['RL109 t-018'])
def run_allopticalNontargets(missing_slmtargets_sz_stim, **kwargs):
    expobj = kwargs['expobj']
    force_redo = True if not hasattr(expobj, 's2p_nontargets') else False
    if hasattr(expobj, 'slmtargets_szboundary_stim'):
        aoutils.run_allopticalAnalysisNontargetsPost4ap(expobj=expobj, normalize_to='pre-stim', do_processing=True,
                                                        to_plot=True, force_redo=force_redo,
                                                        save_plot_suffix=f"{expobj.metainfo['animal prep.']}_{expobj.metainfo['trial']}-post4ap.png")
    else:
        missing_slmtargets_sz_stim.append(f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}")
run_allopticalNontargets(missing_slmtargets_sz_stim=missing_slmtargets_sz_stim)



# %% 1.2) collect average stats for each prep, and summarize into the appropriate data point
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
            pre4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            pre4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

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


    num_suite2p_rois = len(expobj.good_cells)  # this will be the same number for pre and run_post4ap_trials (as they should be the same cells)


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
            post4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            post4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_nontargets_avg[expobj.sig_units][
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
