## script dedicated to code that focuses on analysis re: SLM targets data
# %% IMPORT MODULES AND TRIAL expobj OBJECT
import sys
from typing import Union

from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesQuantificationSLMtargets
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tf
from funcsforprajay import funcs as pj

import alloptical_utils_pj as aoutils
from _utils_ import alloptical_plotting_utils as aoplot

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)
# aoutils.random_plot()
# expobj = Utils.import_expobj(prep='RL109', trial='t-013', verbose=True)
# key = 'f'; exp = 'pre'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")


"##### -------------------- ALL OPTICAL PHOTOSTIM ANALYSIS #############################################################"

# TODO CURRENTLY IN THE PROCESS OF TRANSFERRING ALL OPTICAL PHOTOSTIM SLM TARGETS ANALYSIS TO CLASS BASED MODULE



# %% 5.0) calculate/collect min distance to seizure and responses at each distance
import sys; sys.exit()

response_type = 'dFF (z scored) (interictal)'

no_slmtargets_szboundary_stim = []
@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def run_calculating_min_distance_to_seizure(**kwargs):
    print(f"\t\- collecting responses vs. distance to seizure [5.0-1]")

    expobj = kwargs['expobj']
    if not hasattr(expobj, 'stimsSzLocations'):
        expobj.sz_locations_stims()
    x_ = expobj.calcMinDistanceToSz()
    no_slmtargets_szboundary_stim.append(x_)


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_sz_boundary_location(**kwargs):
    expobj = kwargs['expobj']
    aoplot.plot_sz_boundary_location(expobj)


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def collect_responses_vs_distance_to_seizure_SLMTargets(response_type: str, **kwargs):
    """

    :param response_type: either 'dFF (z scored)' or 'dFF (z scored) (interictal)'
    :param kwargs: must contain expobj as arg key
    """
    print(f"\t\- collecting responses vs. distance to seizure [5.0-2]")
    expobj = kwargs['expobj']

    # # uncomment if need to rerun for a particular expobj....but shouldn't really need to be doing so
    # if not hasattr(expobj, 'responses_SLMtargets_tracedFF'):
    #     expobj.StimSuccessRate_SLMtargets_tracedFF, expobj.hits_SLMtargets_tracedFF, expobj.responses_SLMtargets_tracedFF, expobj.traces_SLMtargets_tracedFF_successes = \
    #         expobj.get_SLMTarget_responses_dff(process='trace dFF', threshold=10, stims_to_use=expobj.stim_start_frames)
    #     print(f'WARNING: {expobj.t_series_name} had to rerun .get_SLMTarget_responses_dff')

    # (re-)make pandas dataframe
    df = pd.DataFrame(columns=['target_id', 'stim_id', 'inorout_sz', 'distance_to_sz', response_type])

    stim_ids = [(idx, stim) for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
    for target in expobj.responses_SLMtargets_tracedFF.index:

        ## z-scoring of SLM targets responses:
        z_scored = expobj.responses_SLMtargets_tracedFF  # initializing z_scored df
        if response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)':
            # set a different slice of stims for different variation of z scoring
            if response_type == 'dFF (z scored)': slice = expobj.responses_SLMtargets_tracedFF.columns  # (z scoring all stims all together from t-series)
            elif response_type == 'dFF (z scored) (interictal)': slice = expobj.stim_idx_outsz  # (z scoring all stims relative TO the interictal stims from t-series)
            __mean = expobj.responses_SLMtargets_tracedFF.loc[target, slice].mean()
            __std = expobj.responses_SLMtargets_tracedFF.loc[target, slice].std(ddof=1)
            # __mean = expobj.responses_SLMtargets_tracedFF.loc[target, :].mean()
            # __std = expobj.responses_SLMtargets_tracedFF.loc[target, :].std(ddof=1)

            __responses = expobj.responses_SLMtargets_tracedFF.loc[target, :]
            z_scored.loc[target, :] = (__responses - __mean) / __std

        for idx, stim in stim_ids:
            if target in expobj.slmtargets_szboundary_stim[stim]: inorout_sz = 'in'
            else: inorout_sz = 'out'

            distance_to_sz = expobj.distance_to_sz['SLM Targets'].loc[target, stim]

            if response_type == 'dFF': response = expobj.responses_SLMtargets_tracedFF.loc[target, idx]
            elif response_type == 'dFF (z scored)' or response_type == 'dFF (z scored) (interictal)': response = z_scored.loc[target, idx]  # z - scoring of SLM targets responses:
            else: raise ValueError('response_type arg must be `dFF` or `dFF (z scored)` or `dFF (z scored) (interictal)`')

            df = df.append({'target_id': target, 'stim_id': stim, 'inorout_sz': inorout_sz, 'distance_to_sz': distance_to_sz,
                            response_type: response}, ignore_index=True)

    expobj.responses_vs_distance_to_seizure_SLMTargets = df

    # convert distances to microns
    expobj.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz_um'] = round(expobj.responses_vs_distance_to_seizure_SLMTargets['distance_to_sz'] / expobj.pix_sz_x, 2)
    expobj.save()


# run_calculating_min_distance_to_seizure(no_slmtargets_szboundary_stim)

collect_responses_vs_distance_to_seizure_SLMTargets(response_type=response_type)



# %% 5.1) PLOT - collect and plot targets responses for stims vs. distance
@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_responses_vs_distance_to_seizure_SLMTargets(response_type=response_type, **kwargs):
    # response_type = 'dFF (z scored)'

    print(f"\t|- plotting responses vs. distance to seizure [5.1-1]")
    expobj = kwargs['expobj']
    fig, ax = plt.subplots(figsize=[8, 3])
    for target in expobj.responses_SLMtargets_tracedFF.index:
        idx_sz_boundary = [idx for idx, stim in enumerate(expobj.stim_start_frames) if stim in expobj.distance_to_sz['SLM Targets'].columns]
        responses = np.array(expobj.responses_SLMtargets_tracedFF.loc[target, idx_sz_boundary])
        distance_to_sz = np.array(expobj.distance_to_sz['SLM Targets'].loc[target, idx_sz_boundary])

        positive_distances = np.where(distance_to_sz > 0)
        negative_distances = np.where(distance_to_sz < 0)

        pj.make_general_scatter(x_list=[distance_to_sz[positive_distances]], y_data=[responses[positive_distances]], fig=fig, ax=ax, colors=['cornflowerblue'], alpha=0.5, s=30, show=False,
                                x_label='distance to sz', y_label=response_type)
        pj.make_general_scatter(x_list=[distance_to_sz[negative_distances]], y_data=[responses[negative_distances]], fig=fig, ax=ax, colors=['tomato'], alpha=0.5, s=30, show=False,
                                x_label='distance to sz', y_label=response_type)

    fig.suptitle(expobj.t_series_name)
    fig.show()



@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_collection_response_distance(response_type=response_type, **kwargs):
    print(f"\t|- plotting a collection of plots measuring responses vs. distance to seizure [5.1-2]")
    expobj = kwargs['expobj']
    # response_type = 'dFF (z scored)'
    if not hasattr(expobj, 'responses_SLMtargets_tracedFF_avg_df'):
        expobj.avgResponseSzStims_SLMtargets(save=True)

    data = expobj.responses_vs_distance_to_seizure_SLMTargets
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

# plot_responses_vs_distance_to_seizure_SLMTargets()
plot_collection_response_distance()



# %% 5.1.1) PLOT - binning and plotting density plot, and smoothing data across the distance to seizure axis, when comparing to responses - represent the distances in percentile space


@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True)
def plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(response_type, positive_distances_only = False, plot=True, **kwargs):

    print(f"\t|- plotting responses vs. distance to seizure")
    expobj = kwargs['expobj']

    data_expobj = np.array([[], []]).T
    for target in expobj.responses_SLMtargets_tracedFF.index:
        indexes = expobj.responses_vs_distance_to_seizure_SLMTargets[expobj.responses_vs_distance_to_seizure_SLMTargets['target_id'] == target].index
        responses = np.array(expobj.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, response_type])
        distance_to_sz = np.asarray(expobj.responses_vs_distance_to_seizure_SLMTargets.loc[indexes, 'distance_to_sz_um'])
        # distance_to_sz_ = np.array(list(expobj.distance_to_sz['SLM Targets'].loc[target, :]))

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

    pj.plot_hist2d(data=data_expobj, bins=bins_num, y_label=response_type, title=expobj.t_series_name, figsize=(4, 2), x_label='distance to seizure (um)',
                   y_lim=[-2,2]) if plot else None


    return data_expobj

data = plot_responses_vs_distance_to_seizure_SLMTargets_2ddensity(response_type=response_type, positive_distances_only = False, plot=False)

def convert_responses_szdistances_percentile_space():
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

data_all, percentiles, responses_sorted, distances_to_sz_sorted, scale_percentile_distances = convert_responses_szdistances_percentile_space()

def plot_density_responses_szdistances(response_type=response_type, data_all=data_all, distances_to_sz_sorted=distances_to_sz_sorted):
    # plotting density plot for all exps, in percentile space (to normalize for excess of data at distances which are closer to zero) - TODO any smoothing?

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

plot_density_responses_szdistances()



# plotting line plot for all datapoints for responses vs. distance to seizure

def plot_lineplot_responses_pctszdistances(percentiles, responses_sorted, response_type=response_type):
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

plot_lineplot_responses_pctszdistances(percentiles, responses_sorted)



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

# # %% 2.2) DATA COLLECTION SLMTargets: organize SLMTargets stim responses - across all appropriate run_pre4ap_trials, run_post4ap_trials trial comparisons - using whole trace dFF responses
# """ doing it in this way so that its easy to use in the response vs. stim times relative to seizure onset code (as this has already been coded up)"""
#
# trials_skip = [
#     'RL108 t-011',
#     'RL109 t-017'  # RL109 t-017 doesn't have sz boundaries yet.. just updated the sz onset/offset's
# ]
#
# trials_run = [
#     'PS11 t-010'
# ]
#
# allopticalResults.outsz_missing = []
# allopticalResults.insz_missing = []
# stim_responses_tracedFF_comparisons_dict = {}
# for i in range(len(allopticalResults.pre_4ap_trials)):
#     prep = allopticalResults.pre_4ap_trials[i][0][:-6]
#     pre4aptrial = allopticalResults.pre_4ap_trials[i][0][-5:]
#     post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:]
#     date = \
#     allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == f"{prep} {pre4aptrial}", 'date'].values[0]
#     print("\n\n\n Starting for loop to make .stim_responses_tracedFF_comparisons_dict -------------------------")
#     print(f"\t{i}, {date}, {prep}, run_pre4ap_trials trial: {pre4aptrial}, run_post4ap_trials trial: {post4aptrial}")
#
#     # skipping some trials that need fixing of the expobj
#     if f"{prep} {pre4aptrial}" not in trials_skip:
#
#         # load up pre-4ap trial
#         print(f'|-- importing {prep} {pre4aptrial} - run_pre4ap_trials trial')
#
#         expobj, experiment = aoutils.import_expobj(trial=pre4aptrial, date=date, prep=prep, verbose=False,
#                                                    do_processing=False)
#         # collect raw Flu data from SLM targets
#         expobj.collect_traces_from_targets(force_redo=False)
#         aoutils.run_alloptical_processing_photostim(expobj, plots=False,
#                                                     force_redo=False)  # REVIEW PROGRESS: run_pre4ap_trials seems to be working fine till here for trace_dFF processing
#
#         df = expobj.responses_SLMtargets_tracedFF.T  # df == stim frame x cells (photostim targets)
#         if len(allopticalResults.pre_4ap_trials[i]) > 1:
#             for j in range(len(allopticalResults.pre_4ap_trials[i]))[1:]:
#                 print(f"\---- {i}, {j}")
#                 # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
#                 prep = allopticalResults.pre_4ap_trials[i][j][:-6]
#                 pre4aptrial_ = allopticalResults.pre_4ap_trials[i][j][-5:]
#                 if f"{prep} {pre4aptrial}" not in trials_skip:
#                     print(f"\------ adding trial to this comparison: {pre4aptrial_} [1.0]")
#                     date = list(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == '%s %s' % (
#                     prep, pre4aptrial_), 'date'])[0]
#
#                     # load up pre-4ap trial
#                     print(f'\------ importing {prep} {pre4aptrial_} - run_pre4ap_trials trial')
#                     expobj, experiment = aoutils.import_expobj(trial=pre4aptrial_, date=date, prep=prep, verbose=False,
#                                                                do_processing=False)
#                     # collect raw Flu data from SLM targets
#                     expobj.collect_traces_from_targets(force_redo=False)
#                     aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)
#
#                     df_ = expobj.responses_SLMtargets_tracedFF.T
#
#                     # append additional dataframe to the first dataframe
#                     df.append(df_, ignore_index=True)
#                 else:
#                     print(f"\------ ***** skipping: {prep} {pre4aptrial_}")
#
#         # accounting for multiple pre/post photostim setup comparisons within each prep
#         if prep not in stim_responses_tracedFF_comparisons_dict.keys():
#             stim_responses_tracedFF_comparisons_dict[prep] = {}
#             comparison_number = 1
#         else:
#             comparison_number = len(stim_responses_tracedFF_comparisons_dict[prep]) + 1
#
#         # stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}'] = {'pre-4ap': {}, 'post-4ap': {}, 'in sz': {}}  # initialize dict for saving responses
#         stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}'] = {
#             'pre-4ap': {}}  # initialize dict for saving responses
#         stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['pre-4ap'] = df
#
#         pre_4ap_df = df
#
#
#     else:
#         print(f"|-- skipping: {prep} run_pre4ap_trials trial {pre4aptrial}")
#
#     ##### POST-4ap trials - OUT OF SZ PHOTOSTIMS
#     print(f'TEST 1.1 - working on {prep}, run_post4ap_trials trial {post4aptrial}')
#
#     # skipping some trials that need fixing of the expobj
#     if f"{prep} {post4aptrial}" not in trials_skip:
#
#         # load up post-4ap trial and stim responses
#         print(f'|-- importing {prep} {post4aptrial} - run_post4ap_trials trial')
#         expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False,
#                                                    do_processing=False)
#         # collect raw Flu data from SLM targets
#         expobj.collect_traces_from_targets(force_redo=False)
#         aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)
#
#         if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
#             df = expobj.responses_SLMtargets_tracedFF_outsz.T
#
#             if len(allopticalResults.post_4ap_trials[i]) > 1:
#                 for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
#                     print(f"\---- {i}, {j}")
#                     # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
#                     prep = allopticalResults.post_4ap_trials[i][j][:-6]
#                     post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
#                     if f"{prep} {post4aptrial_}" not in trials_skip:
#                         print(f"\------ adding trial to this comparison: {post4aptrial} [1.1]")
#                         date = list(allopticalResults.metainfo.loc[
#                                         allopticalResults.metainfo['prep_trial'] == '%s %s' % (
#                                         prep, pre4aptrial), 'date'])[0]
#
#                         # load up post-4ap trial and stim responses
#                         print(f'\------ importing {prep} {post4aptrial_} - run_post4ap_trials trial')
#                         expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
#                                                                    verbose=False, do_processing=False)
#                         # collect raw Flu data from SLM targets
#                         expobj.collect_traces_from_targets(force_redo=False)
#                         aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)
#
#                         if hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
#                             df_ = expobj.responses_SLMtargets_tracedFF_outsz.T
#                             # append additional dataframe to the first dataframe
#                             df.append(df_, ignore_index=True)
#                         else:
#                             print(
#                                 '\------ **** 2 need to run collecting outsz responses SLMtargets attr for %s %s ****' % (
#                                 post4aptrial_, prep))
#                             allopticalResults.outsz_missing.append('%s %s' % (post4aptrial_, prep))
#                     else:
#                         print(f"\---- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial_}")
#
#             stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']['post-4ap'] = df
#
#         else:
#             print('\-- **** need to run collecting outsz responses SLMtargets attr for %s %s **** [1]' % (
#             post4aptrial, prep))
#             allopticalResults.outsz_missing.append('%s %s' % (post4aptrial, prep))
#
#         ##### POST-4ap trials - IN SZ PHOTOSTIMS - only PENUMBRA cells
#         # post4aptrial = allopticalResults.post_4ap_trials[i][0][-5:] -- same as run_post4ap_trials OUTSZ for loop one above
#
#         # skipping some trials that need fixing of the expobj
#         # if f"{prep} {post4aptrial}" not in skip_trials:
#         #     print(f'TEST 1.2 - working on {prep} {post4aptrial}')
#
#         # using the same skip statement as in the main for loop here
#
#         # load up post-4ap trial and stim responses
#         expobj, experiment = aoutils.import_expobj(trial=post4aptrial, date=date, prep=prep, verbose=False,
#                                                    do_processing=False)
#         # collect raw Flu data from SLM targets
#         expobj.collect_traces_from_targets(force_redo=False)
#         aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)
#
#         if hasattr(expobj, 'slmtargets_szboundary_stim'):
#             if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
#                 df = expobj.responses_SLMtargets_tracedFF_insz.T
#
#                 # switch to NA for stims for cells which are classified in the sz
#                 # collect stim responses with stims excluded as necessary
#                 for target in df.columns:
#                     # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
#                     for stim in list(expobj.slmtargets_szboundary_stim.keys()):
#                         if target in expobj.slmtargets_szboundary_stim[stim]:
#                             df.loc[expobj.stim_start_frames.index(stim)][target] = np.nan
#
#                     # responses = [expobj.responses_SLMtargets_tracedFF_insz.loc[col][expobj.stim_start_frames.index(stim)] for stim in expobj.stims_in_sz if
#                     #              col not in expobj.slmtargets_szboundary_stim[stim]]
#                     # targets_avgresponses_exclude_stims_sz[row] = np.mean(responses)
#
#                 if len(allopticalResults.post_4ap_trials[i]) > 1:
#                     for j in range(len(allopticalResults.post_4ap_trials[i]))[1:]:
#                         print(f"|-- {i}, {j}")
#                         # if there are multiple trials for this comparison then append stim frames for repeat trials to the dataframe
#                         prep = allopticalResults.post_4ap_trials[i][j][:-6]
#                         post4aptrial_ = allopticalResults.post_4ap_trials[i][j][-5:]
#                         if f"{prep} {post4aptrial_}" not in trials_skip:
#                             print(f"{post4aptrial} [1.2]")
#                             date = list(allopticalResults.metainfo.loc[
#                                             allopticalResults.metainfo['prep_trial'] == '%s %s' % (
#                                             prep, pre4aptrial), 'date'])[0]
#
#                             # load up post-4ap trial and stim responses
#                             expobj, experiment = aoutils.import_expobj(trial=post4aptrial_, date=date, prep=prep,
#                                                                        verbose=False, do_processing=False)
#                             # collect raw Flu data from SLM targets
#                             expobj.collect_traces_from_targets(force_redo=False)
#                             aoutils.run_alloptical_processing_photostim(expobj, plots=False, force_redo=False)
#
#                             if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
#                                 df_ = expobj.responses_SLMtargets_tracedFF_insz.T
#
#                                 # append additional dataframe to the first dataframe
#                                 # switch to NA for stims for cells which are classified in the sz
#                                 # collect stim responses with stims excluded as necessary
#                                 for target in df.columns:
#                                     # stims = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
#                                     for stim in list(expobj.slmtargets_szboundary_stim.keys()):
#                                         if target in expobj.slmtargets_szboundary_stim[stim]:
#                                             df_.loc[expobj.stim_start_frames.index(stim)][target] = np.nan
#
#                                 df.append(df_, ignore_index=True)
#                             else:
#                                 print(
#                                     '**** need to run collecting in sz responses SLMtargets attr for %s %s **** [4]' % (
#                                     post4aptrial_, prep))
#                                 allopticalResults.insz_missing.append('%s %s' % (post4aptrial_, prep))
#                         else:
#                             print(f"\-- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial_}")
#
#                 stim_responses_tracedFF_comparisons_dict[prep][f"{comparison_number}"]['in sz'] = df
#             else:
#                 print('**** need to run collecting insz responses SLMtargets attr for %s %s **** [4]' % (
#                 post4aptrial, prep))
#                 allopticalResults.insz_missing.append('%s %s' % (post4aptrial, prep))
#         else:
#             print(f"**** need to run collecting slmtargets_szboundary_stim for {prep} {post4aptrial} [5]")
#
#     else:
#         print(f"\-- ***** skipping: {prep} run_post4ap_trials trial {post4aptrial}")
#         if not hasattr(expobj, 'responses_SLMtargets_tracedFF_outsz'):
#             print(
#                 f'\-- **** need to run collecting outsz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [1]')
#
#         if not hasattr(expobj, 'slmtargets_szboundary_stim'):
#             print(
#                 f'**** need to run collecting insz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [2]')
#         if hasattr(expobj, 'responses_SLMtargets_tracedFF_insz'):
#             print(
#                 f'**** need to run collecting in sz responses SLMtargets attr for run_post4ap_trials trial {post4aptrial}, {prep} **** [3]')
#
#     ## switch out the comparison_number to something more readable
#     new_key = f"{pre4aptrial} vs. {post4aptrial}"
#     stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep].pop(
#         f'{comparison_number}')
#     # stim_responses_tracedFF_comparisons_dict[prep][new_key] = stim_responses_tracedFF_comparisons_dict[prep][f'{comparison_number}']
#
#     # save to: allopticalResults.stim_responses_tracedFF_comparisons
#     allopticalResults.stim_responses_tracedFF_comparisons = stim_responses_tracedFF_comparisons_dict
#     allopticalResults.save()


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


# %% 3.2-dc) TODO DATA COLLECTION - COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ -  need to investigate how REAL the negative going values are in the FAILURES responses in POST4AP trials

# collecting the response magnitudes of FAILURES stims

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





# %% 6.0-main) avg STA responses in 200um space around photostim targets - compare diff between pre vs. post4ap (interictal, and ictal)

# for i in ['pre', 'post']:
key = 'l'; exp = 'post'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, run_trials=['PS04 t-012', 'PS04 t-017', 'PS04 t-018'])
def collect_responses_around_slmtarget(do_plot=True, **kwargs):
    assert 'expobj' in kwargs.keys(), "need to provide `expobj` as key word arg for function to complete"
    expobj = kwargs['expobj']
    # key = 'l';
    # exp = 'pre';
    # expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")

    # read in frames of interest from the reg_tiff, extract data around each SLM target as 2d x frames array, put into dictionary with SLM target as key


    # read in frames of interest
    frames_of_interest = expobj.stims_out_sz if hasattr(expobj, 'stims_out_sz') else expobj.stim_start_frames
    # expobj.stitch_reg_tiffs(force_crop=True, do_stack=True) if not os.path.exists(expobj.reg_tif_crop_path) else None

    print(f"|- reading in registered TIFF for expobj from: {expobj.reg_tif_crop_path}")
    try:
        tiff_arr = tf.imread(expobj.reg_tif_crop_path, key=range(expobj.n_frames))
    except:
        expobj.stitch_reg_tiffs(force_crop=True, do_stack=True)
        tiff_arr = tf.imread(expobj.reg_tif_crop_path, key=range(expobj.n_frames))

    data_ = {}
    dist_um = 200
    dist_pix_half = int((dist_um / expobj.pix_sz_x) / 2)

    # collecting responses for all targets at all stims for the experiment
    print(f"|- collecting traces at distance: {dist_um} um around target coordinate")
    for idx, coord in enumerate(expobj.target_coords_all):
        ## make a slice object
        s = np.s_[:, int(coord[1]) - dist_pix_half: int(coord[1]) + dist_pix_half,
            int(coord[0]) - dist_pix_half: int(coord[0]) + dist_pix_half]

        ## use the slice object to collect Flu trace
        target_area_trace = tiff_arr[s]

        # dFF normalize
        mean = np.mean(target_area_trace)
        target_area_trace_dff = (target_area_trace - mean) / mean  # dFF

        frames_responses = np.empty(shape=(len(frames_of_interest), target_area_trace_dff.shape[1], target_area_trace_dff.shape[2]))
        for idx, stim_frame in enumerate(frames_of_interest):
            pre_slice = target_area_trace_dff[stim_frame - expobj.pre_stim_response_frames_window : stim_frame, :, :]
            post_slice = target_area_trace_dff[stim_frame + expobj.stim_duration_frames: stim_frame + expobj.stim_duration_frames + expobj.post_stim_response_frames_window, :, :]
            dF = np.mean(post_slice, axis=0) - np.mean(pre_slice, axis=0)
            frames_responses[idx] = dF

        data_[idx] = frames_responses
    print(f"|- collected data from {len(expobj.target_coords_all)} SLM target coords and {len(frames_of_interest)} stim frames")
    data_['data_collected_at_distance_um'] = dist_um
    expobj.SLMtarget_areas_responses = data_
    expobj.save()


    # second approach: collecting responses around targets using hits stim trials only from this expobj
    locs = np.where(expobj.hits_SLMtargets == 1)
    # for idx, stim_idx in zip(locs[0], locs[1]):
    for cell_idx, coord in enumerate(expobj.target_coords_all):
        stim_idxes = locs[1][np.where(locs[0] == cell_idx)]
        stim_frames_list = [expobj.stim_start_frames[stim_idx] for stim_idx in stim_idxes]
        ## make a slice object
        s = np.s_[:, int(coord[1]) - dist_pix_half: int(coord[1]) + dist_pix_half,
            int(coord[0]) - dist_pix_half: int(coord[0]) + dist_pix_half]

        ## use the slice object to collect Flu trace
        target_area_trace = tiff_arr[s]

        # dFF normalize
        mean = np.mean(target_area_trace)
        target_area_trace_dff = (target_area_trace - mean) / mean  # dFF

        frames_responses = np.empty(shape=(len(stim_frames_list), target_area_trace_dff.shape[1], target_area_trace_dff.shape[2]))
        for idx, stim_frame in enumerate(stim_frames_list):
            pre_slice = target_area_trace_dff[stim_frame - expobj.pre_stim_response_frames_window : stim_frame, :, :]
            post_slice = target_area_trace_dff[stim_frame + expobj.stim_duration_frames: stim_frame + expobj.stim_duration_frames + expobj.post_stim_response_frames_window, :, :]
            dF = np.mean(post_slice, axis=0) - np.mean(pre_slice, axis=0)
            frames_responses[idx] = dF

        data_[cell_idx] = frames_responses
    print(f"|- collected data from {len(expobj.target_coords_all)} SLM target coords and ONLY HITS stim frames for each target")
    data_['data_collected_at_distance_um'] = dist_um
    expobj.SLMtarget_areas_responses_hitsonly = data_
    expobj.save()


    if do_plot:
        # plot heatmaps
        fig, axs = plt.subplots(nrows=11, ncols=5, figsize=(15, 33))
        print(f"|- making avg STA plots for {len(expobj.target_coords_all)} SLM target coords ... ", end='\r')
        counter = 0
        for idx, coord in enumerate(expobj.target_coords_all):
            col = counter % 5
            row = counter // 5
            mean_ = np.mean(expobj.SLMtarget_areas_responses[idx], axis=0)  # mean across all stim frames
            hmap = axs[row, col].imshow(mean_, vmin=0, vmax=0.2)
            counter += 1
        colorbar = fig.colorbar(hmap, ax=axs[row, col], fraction=0.046, pad=0.04)
        fig.tight_layout(pad=1.3)
        fig.suptitle(f"{expobj.t_series_name} - {len(expobj.target_coords_all)} targets - {expobj.exptype}", y=0.995)
        fig.show()
        print(f"|- making avg STA plots for {len(expobj.target_coords_all)} SLM target coords ... DONE", end='\r')

collect_responses_around_slmtarget(do_plot=False)


# %% 6.1-plot) Ca responses around SLM targets

for key in list(allopticalResults.trial_maps['pre'].keys())[1:]:

    print(f"\n|- Making plot for key: {key}")

    exp = 'pre'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")
    __means = []
    for idx, coord in enumerate(expobj.target_coords_all):
        arr = np.mean(expobj.SLMtarget_areas_responses_hitsonly[idx], axis=0)
        if arr.shape[0] == arr.shape[1]:
        # print(expobj.SLMtarget_areas_responses[idx].shape)
            __means.append(arr)  # mean across all stim frames
    assert len(__means) > 1, f'no frames in __means for {expobj.t_series_name}'
    pre4ap_mean = np.mean(np.asarray(__means), axis=0)

    exp = 'post'; expobj, experiment = aoutils.import_expobj(aoresults_map_id=f"{exp} {key}.0")
    __means = []
    for idx, coord in enumerate(expobj.target_coords_all):
        arr = np.mean(expobj.SLMtarget_areas_responses_hitsonly[idx], axis=0)
        if arr.shape[0] == arr.shape[1]:
        # print(expobj.SLMtarget_areas_responses[idx].shape)
            __means.append(arr)  # mean across all stim frames
    assert len(__means) > 1, f'no frames in __means for {expobj.t_series_name}'
    post4ap_mean = np.mean(np.asarray(__means), axis=0)

    # plot heatmaps
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(8, 3.1))
    print(f"|- making pre and post4ap compare response plots", end='\r')
    aoplot.plot_sz_boundary_location(expobj, fig=fig, ax=axs[0], title='')
    axs[1].imshow(pre4ap_mean, vmin=0, vmax=0.3)
    axs[1].set_title('pre4ap')
    im = axs[2].imshow(post4ap_mean, vmin=0, vmax=0.3)
    axs[2].set_title('post4ap')
    for ax in axs[1:]:
        xticks = ax.get_xticks()  # pixel space
        # labels = [int(x_ * expobj.pix_sz_x - expobj.SLMtarget_areas_responses['data_collected_at_distance_um'] / 2) for x_ in xticks]
        new_xticks_labels = np.arange(0, expobj.SLMtarget_areas_responses['data_collected_at_distance_um'], 50)[1:]
        new_xticks_labels_adjusted = [int(x_ - expobj.SLMtarget_areas_responses['data_collected_at_distance_um'] / 2) for x_ in new_xticks_labels]
        new_xticks = [x_ // expobj.pix_sz_x for x_ in new_xticks_labels]
        ax.set_xticks(new_xticks)
        ax.set_xticklabels(new_xticks_labels_adjusted)

        ax.set_yticks([])
    axs[1].set_xlabel('distance to SLM target coordinate (um)')

    fig.tight_layout(pad=2.5)
    fig.suptitle(f"{expobj.t_series_name} - {len(expobj.target_coords_all)} targets - {expobj.exptype}", y=0.995)
    fig.show()


# TODO need to figure out how to exclude targets cells from within the plotting zone (atleast for quantification) -- go through the s2p ROIs followers responses
# collect only for success stims - done







# %% archive-1) adding slm targets responses to alloptical results allopticalResults.slmtargets_stim_responses


animal_prep = 'PS07'
date = '2021-01-19'
# trial = 't-009'

pre4ap_trials = ['t-007', 't-008', 't-009']
post4ap_trials = ['t-011', 't-016', 't-017']

# save_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
#     date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
#
# expobj, _ = aoutils.import_expobj(save_path=save_path)

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
