## script dedicated to code that focuses on analysis re: SLM targets data

# %% IMPORT MODULES AND TRIAL expobj OBJECT
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

from skimage import draw

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013', verbose=True)

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

trials_skip = [
    'RL108 t-011',
    'RL109 t-017',  # RL109 t-017 doesn't have sz boundaries yet.. just updated the sz onset/offset's
    'RL109 t-020',  # RL109 t-020 doesn't have sz boundaries yet..
]


# %% 5.0-dc) COLLECT and PLOT targets responses for stims vs. distance (starting with old code)- top priority right now

x = []
@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def run_calculating_min_distance_to_seizure(**kwargs):
    expobj = kwargs['expobj']
    x_ = expobj.calcMinDistanceToSz()
    x.append(x_)
run_calculating_min_distance_to_seizure(x)


# %%


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    expobj = kwargs['expobj']
    fig, ax = plt.subplots(figsize=[3, 3])
    for target in expobj.responses_SLMtargets_tracedFF.index:
        idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        responses = expobj.responses_SLMtargets_tracedFF.loc[target, idx_sz_boundary]
        distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, :]

        pj.make_general_scatter(x_list=[distance_to_sz], y_data=[responses], fig=fig, ax=ax, colors=['cornflowerblue'], alpha=0.5, s=30, show=False,
                                x_label=['distance to sz'], y_label=['response delta (trace dFF)'])
    fig.suptitle(expobj.t_series_name)
    fig.show()

plot_responses_vs_distance_to_seizure_SLMTargets()

# %%

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def collect_responses_vs_distance_to_seizure_SLMTargets(**kwargs):
    expobj = kwargs['expobj']

    # make pandas dataframe
    df = pd.DataFrame(columns=['cells', 'stim_id', 'inorout_sz', 'distance_to_sz', 'response_to_sz'])

    responses_all_cells_all_stims = []
    distance_to_sz_all_cells_all_stims = []
    for target in expobj.responses_SLMtargets_tracedFF.index:
        # idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        inorout_sz = []
        for idx, stim in stim_ids:
            if target in expobj.slmtargets_szboundary_stim[stim]: inorout_sz = 'in'
            else: inorout_sz = 'out'
            distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]
            responseslist = expobj.responses_SLMtargets_tracedFF.loc[target, idx]
            df = df.append({'cells': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                            'response_to_sz': responseslist}, ignore_index=True)

        # targets = [target]*len(stim_ids)

    expobj.responses_vs_distance_to_seizure_SLMTargets = df

    expobj.save()


collect_responses_vs_distance_to_seizure_SLMTargets()


key = 'e'
j = 0
exp = 'post'
expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.{j}")

data = expobj.responses_vs_distance_to_seizure_SLMTargets
g = sns.catplot(x="inorout_sz", y="distance_to_sz", hue="inorout_sz", data=data)

plt.show()



## archived:

        # ax.scatter(x=distance_to_sz, y=responses, color='cornflowerblue',
        #             alpha=0.5, s=30,
        #             zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times

    # ax.set_xlabel('distance to seizure front (pixels)')
    # ax.set_ylabel('response magnitude')
    # ax.set_title('')
    # fig.tight_layout(pad=1.3)
    # fig.show()
    #
    #
    # # calculate linear regression line
    # ax.plot(range(int(min(distance_to_sz)), int(max(distance_to_sz))),
    #          np.poly1d(np.polyfit(distance_to_sz, responses, 1))(
    #              range(int(min(distance_to_sz)), int(max(distance_to_sz)))),
    #          color='black')
    #
    # ax.scatter(x=distance_to_sz, y=responses, color='cornflowerblue',
    #             alpha=0.5, s=16,
    #             zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times
    # ax1.scatter(x=distance_to_sz_, y=responses_, color='firebrick',
    #             alpha=0.5, s=16,
    #             zorder=0)  # use cmap correlated to distance from seizure to define colors of each target at each individual stim times




# %%

# # plot response magnitude vs. distance
# for i, stim_frame in enumerate(expobj.slmtargets_szboundary_stim):
#     target = 0
#     # calculate the min distance of slm target to s2p cells classified inside of sz boundary at the current stim
#     targetsInSz = expobj.slmtargets_szboundary_stim[stim_frame]
#     target_coord = expobj.target_coords_all[target]
#     min_distance = 1200
#     for j, _ in enumerate(targetsInSz):
#         dist = pj.calc_distance_2points(target_coord, tuple(expobj.stat[j]['med']))  # distance in pixels
#         if dist < min_distance:
#             min_distance = dist

fig1, ax1 = plt.subplots(figsize=[5, 5])
responses = []
distance_to_sz = []
responses_ = []
distance_to_sz_ = []
for target in expobj.responses_SLMtargets_dfprestimf.keys():
    mean_response = np.mean(expobj.responses_SLMtargets_dfprestimf[target])
    target_coord = expobj.target_coords_all[target]
    # print(mean_response)

    # calculate response magnitude at each stim time for selected target
    for i, _ in enumerate(expobj.stim_start_frames):
        # the response magnitude of the current SLM target at the current stim time (relative to the mean of the responses of the target over this trial)
        response = expobj.responses_SLMtargets_dfprestimf[target][i] / mean_response  # changed to division by mean response instead of substracting
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


# %% 3.1) DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ


## collecting the response magnitudes of success stims

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def collect_response_mag_successes_deltatracedFF(**kwargs):
    expobj = kwargs['expobj']
    exp_prep = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"

    # PRE4AP
    if 'pre' in expobj.metainfo['exptype']:
        success_responses = expobj.hits_SLMtargets_tracedFF * expobj.responses_SLMtargets_tracedFF
        success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response (hits, all targets)'] = success_responses.mean()
        print(f"\tpre4ap hits mean: {success_responses.mean()}")

    elif 'post' in expobj.metainfo['exptype']:
        # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
        # OUTSZ
        if expobj.stims_out_sz:
            success_responses = expobj.hits_SLMtargets_tracedFF_outsz * expobj.responses_SLMtargets_tracedFF_outsz
            success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response outsz (hits, all targets)'] = success_responses.mean()
            print(f"\toutsz hits mean: {success_responses.mean()}")

        # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
        # INSZ
        if expobj.stims_in_sz:
            success_responses = expobj.hits_SLMtargets_tracedFF_insz * expobj.responses_SLMtargets_tracedFF_insz
            success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response insz (hits, all targets)'] = success_responses.mean()
            print(f"\tinsz hits mean: {success_responses.mean()}")


    return None


res = collect_response_mag_successes_deltatracedFF()

allopticalResults.save()


# %% 3.2-dc)  TODO DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

## - need to investigate how REAL the negative going values are in the FAILURES responses in POST4AP trials

## collecting the response magnitudes of FAILURES stims

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def collect_response_mag_failures_deltatracedFF(**kwargs):
    expobj = kwargs['expobj']
    exp_prep = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"

    # PRE4AP
    if 'pre' in expobj.metainfo['exptype']:
        inverse = (expobj.hits_SLMtargets_tracedFF - 1) * -1
        failures_responses = inverse * expobj.responses_SLMtargets_tracedFF
        failures_responses = failures_responses.replace(0, np.NaN).mean(axis=1)
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                            'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response (misses, all targets)'] = failures_responses.mean()
        print(f"\tpre4ap misses mean: {failures_responses.mean()}")

    elif 'post' in expobj.metainfo['exptype']:
        # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
        # OUTSZ
        if expobj.stims_out_sz:
            inverse = (expobj.hits_SLMtargets_tracedFF_outsz - 1) * -1
            failures_responses = inverse * expobj.responses_SLMtargets_tracedFF_outsz
            failures_responses = failures_responses.replace(0, np.NaN).mean(axis=1)
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response outsz (misses, all targets)'] = failures_responses.mean()
            print(f"\toutsz misses mean: {failures_responses.mean()}")

        # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
        # INSZ
        if expobj.stims_in_sz:
            inverse = (expobj.hits_SLMtargets_tracedFF_insz - 1) * -1
            failures_responses = inverse * expobj.responses_SLMtargets_tracedFF_insz
            failures_responses = failures_responses.replace(0, np.NaN).mean(axis=1)
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                'prep_trial'] == exp_prep, 'mean delta(trace_dFF) response insz (misses, all targets)'] = failures_responses.mean()
            print(f"\tinsz misses mean: {failures_responses.mean()}")


    expobj.save()

collect_response_mag_failures_deltatracedFF()

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

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def add_slmtargets_responses_tracedFF(**kwargs):
    print("\t|- adding slm targets trace dFF responses to allopticalResults.slmtargets_stim_responses")
    print(f"\n{kwargs}")
    expobj = kwargs['expobj'] if 'expobj' in kwargs.keys() else KeyError('need to provide expobj as keyword argument')

    if 'pre' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets)'] = dFprestimF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF.loc[i, :]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

        print(f"\t|- {prep_trial}: delta trace dFF response: {delta_trace_dFF_response:.2f}, reliability: {reliability:.2f},  dFprestimF_response: {dFprestimF_response:.2f}")

    elif 'post' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf_outsz[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf_outsz.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets)'] = dFprestimF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF_outsz.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF_outsz.loc[i, :]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

        print(f"\t|- {prep_trial} (outsz): delta trace dFF response: {delta_trace_dFF_response:.2f}, reliability: {reliability:.2f},  dFprestimF_response: {dFprestimF_response:.2f}")

        if expobj.stims_in_sz:
            prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"

            dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf_insz[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets) insz'] = dFstdF_response

            dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf_insz.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets) insz'] = dFprestimF_response

            reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_insz.values()))
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF)) insz'] = reliability

            delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF_insz.loc[i, :]) for i in range(expobj.n_targets_total)]])
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets) insz'] = delta_trace_dFF_response

            print(f"\t|- {prep_trial}: delta trace dFF response (in sz): {delta_trace_dFF_response:.2f}, reliability (in sz): {reliability:.2f},  dFprestimF_response (in sz): {dFprestimF_response:.2f}")

add_slmtargets_responses_tracedFF()

allopticalResults.slmtargets_stim_responses
allopticalResults.save()

## check allopticalResults.slmtargets_stim_responses
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.post_4ap_trials))]['mean response (delta(trace_dFF) all targets)']
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.post_4ap_trials))]['mean response (delta(trace_dFF) all targets) insz']
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.pre_4ap_trials))]['mean response (delta(trace_dFF) all targets)']


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

        df = expobj.responses_SLMtargets_dfprestimf.T  # df == stim frame x cells (photostim targets)
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
                    df_ = expobj.responses_SLMtargets_dfprestimf.T

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
        # if f"{prep} {post4aptrial}" not in skip_trials:
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
    allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == f"{prep} {pre4aptrial}", 'date'].values[0]
    print("\n\n\n Starting for loop to make .stim_responses_tracedFF_comparisons_dict -------------------------")
    print(f"\t{i}, {date}, {prep}, run_pre4ap_trials trial: {pre4aptrial}, run_post4ap_trials trial: {post4aptrial}")

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
        # if f"{prep} {post4aptrial}" not in skip_trials:
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

# %% 4.1) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: dF/prestimF - for loop over all experiments to collect responses in terms of sz onset time

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

# %% 4.2) DATA COLLECTION SLMTargets - absolute stim responses vs. TIME to seizure onset - responses: delta(dFF) from whole trace - for loop over all experiments to collect responses in terms of sz onset time

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

    allopticalResults.stim_relative_szonset_vs_deltatracedFFresponse_alltargets_atstim = stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim
    print(
        f"\tlength of allopticalResults.stim_relative_szonset_vs_avg_dFFresponse_alltargets_atstim dict: {len(allopticalResults.stim_relative_szonset_vs_deltatracedFFresponse_alltargets_atstim.keys())}")
    allopticalResults.save()


# %% 6.0-main) avg responses in 200um space around photostim targets - pre vs. run_post4ap_trials




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
            [[np.mean(expobj.responses_SLMtargets_dfprestimf[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(
            list(expobj.StimSuccessRate_SLMtargets.values()))

    allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dFF all targets)'] = np.nan
    counter += 1
