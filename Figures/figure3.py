import numpy as np
from matplotlib import pyplot as plt

from _alloptical_utils import run_for_loop_across_exps
from _analysis_._ClassPhotostimAnalysisSlmTargets import plot__avg_photostim_dff_allexps
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_.io import import_expobj
from funcsforprajay import plotting as pplot

import xml.etree.ElementTree as ET



# %% D) BAR PLOT OF AVG PHOTOSTIMULATION FOV RAW FLU ACROSS CONDITIONS

# 1.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion
@run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, allow_rerun=1)
def collect_avg_prestimf_baseline(**kwargs):
    expobj: alloptical = kwargs['expobj']
    fov_flu = np.mean(expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu'])
    return fov_flu

@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
def collect_avg_prestimf_interictal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    fov_flu = np.mean(expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu'][expobj.stim_idx_outsz])
    return fov_flu

@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
def collect_avg_prestimf_ictal(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    fov_flu = np.mean(expobj.PhotostimResponsesSLMTargets.adata.var['pre_stim_FOV_Flu'][expobj.stim_idx_insz])
    return fov_flu

baseline_prestimf = collect_avg_prestimf_baseline()
interictal_prestimf = collect_avg_prestimf_interictal()
ictal_prestimf = collect_avg_prestimf_ictal()


# %%

pplot.plot_bar_with_points(data=[baseline_prestimf, interictal_prestimf, ictal_prestimf],
                           bar = False, title='avg prestim F - targets',
                           x_tick_labels=['Baseline', 'Interictal', 'Ictal'],
                           colors=['royalblue', 'forestgreen', 'purple'], figsize=(4,4), y_label='Fluorescence (a.u.)',
                           ylims=[0, 2000], alpha=0.7)



# %% E) BAR PLOT OF AVG PHOTOSTIMULATION RESPONSE OF TARGETS ACROSS CONDITIONS

# 1.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion
@run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, allow_rerun=1)
def collect_avg_photostim_response_baseline(**kwargs):
    expobj: alloptical = kwargs['expobj']

    avg_response = np.mean(expobj.PhotostimResponsesSLMTargets.adata.X, axis=1)
    return np.mean(avg_response)

@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
def collect_avg_photostim_response_interictal(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    interictal_avg_response = np.mean(expobj.PhotostimResponsesSLMTargets.adata.X[:, expobj.stim_idx_outsz], axis=1)
    return np.mean(interictal_avg_response)

@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=True, allow_rerun=1)
def collect_avg_photostim_response_ictal(**kwargs):
    expobj: Post4ap = kwargs['expobj']

    ictal_avg_response = np.mean(expobj.PhotostimResponsesSLMTargets.adata.X[:, expobj.stim_idx_insz], axis=1)
    return np.mean(ictal_avg_response)

baseline_responses = collect_avg_photostim_response_baseline()
interictal_responses = collect_avg_photostim_response_interictal()
ictal_responses = collect_avg_photostim_response_ictal()

pplot.plot_bar_with_points(data=[baseline_responses, interictal_responses, ictal_responses],
                           bar = False, title='avg photostim responses - targets',
                           x_tick_labels=['Baseline', 'Interictal', 'Ictal'],
                           colors=['royalblue', 'forestgreen', 'purple'], figsize=(4,4), y_label='dFF',
                           ylims=[-19, 90], alpha=0.7)





# %% C) GRAND AVERAGE PLOT OF TARGETS AND NONTARGETS
plot__avg_photostim_dff_allexps()

# 1.1) plot the first sz frame for each seizure from each expprep, label with the time delay to sz invasion
@run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=False, allow_rerun=True)
def collect_avg_photostim_traces(**kwargs):
    expobj: alloptical = kwargs['expobj']

    pre_sec = expobj.PhotostimAnalysisSlmTargets.pre_stim_sec
    post_sec = expobj.PhotostimAnalysisSlmTargets.post_stim_sec
    pre_fr = expobj.PhotostimAnalysisSlmTargets.pre_stim_fr
    post_fr = expobj.PhotostimAnalysisSlmTargets.post_stim_fr

    stim_dur = 0.5  # for making adjusted traces below


    ## targets
    targets_traces = expobj.SLMTargets_stims_dff
    target_traces_adjusted = []
    for trace_snippets in targets_traces:
        trace = np.mean(trace_snippets, axis=0)
        pre_stim_trace = trace[:pre_fr]
        post_stim_trace = trace[-post_fr:]
        stim_trace = [0] * int(expobj.fps * stim_dur)  # frames

        new_trace = np.concatenate([pre_stim_trace, stim_trace, post_stim_trace])
        if expobj.fps > 20:
            new_trace = new_trace[::2][:67]

        target_traces_adjusted.append(new_trace)

    avg_photostim_trace_targets = np.mean(target_traces_adjusted, axis=0)


    ## fakestims
    targets_traces = expobj.fake_SLMTargets_tracedFF_stims_dff
    fakestims_target_traces_adjusted = []
    for trace_snippets in targets_traces:
        trace = np.mean(trace_snippets, axis=0)
        pre_stim_trace = trace[:pre_fr]
        post_stim_trace = trace[-post_fr:]
        stim_trace = [0] * int(expobj.fps * stim_dur)  # frames

        new_trace = np.concatenate([pre_stim_trace, stim_trace, post_stim_trace])
        if expobj.fps > 20:
            new_trace = new_trace[::2][:67]

        fakestims_target_traces_adjusted.append(new_trace)

    avg_fakestim_trace_targets = np.mean(fakestims_target_traces_adjusted, axis=0)


    # make corresponding time array
    time_arr = np.linspace(-pre_sec, post_sec + stim_dur, len(avg_photostim_trace_targets))

    # # plotting
    # plt.plot(time_arr, avg_photostim_trace)
    # plt.show()

    print('length of traces; ', len(time_arr))

    return target_traces_adjusted, fakestims_target_traces_adjusted, time_arr

func_collector = collect_avg_photostim_traces()

targets_average_traces = []
fakestim_targets_average_traces = []
for results in func_collector:
    traces = results[0]
    for trace in traces:
        targets_average_traces.append(trace)
    traces = results[1]
    for trace in traces:
        fakestim_targets_average_traces.append(trace)

time_arr = func_collector[0][2]

# %%
fig, ax = plt.subplots(figsize=(3, 4))

# targets
avg_ = np.mean(targets_average_traces, axis=0)
std_ = np.std(targets_average_traces, axis=0, ddof=1)
ax.plot(time_arr, avg_, color='forestgreen', lw=3.5)
ax.fill_between(x=time_arr, y1=avg_ + std_, y2=avg_ - std_, alpha=0.3, zorder=2, color='lightgreen')

# # fakestims - targets
# avg_ = np.mean(fakestim_targets_average_traces, axis=0)
# std_ = np.std(fakestim_targets_average_traces, axis=0, ddof=1)
# ax.plot(time_arr, avg_, color='black', lw=1.5)
# ax.fill_between(x=time_arr, y1=avg_ + std_, y2=avg_ - std_, alpha=0.3, zorder=2, color='gray')

# span over stim frames
stim_ = np.where(std_ == 0)[0]
ax.axvspan(time_arr[stim_[0]-1], time_arr[stim_[-1] + 2], color='hotpink', zorder = 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
ax.set_ylabel('dFF')
ax.set_xlabel('Time (secs) rel. to stim')
fig.suptitle('grand average across all cells')
fig.tight_layout(pad=0.5)
fig.show()


# %% A) getting num pixels for scale bar

# using RL108 for example gcamp c1v1 image

def _getPVStateShard(path, key):
    '''
    Used in function PV metadata below
    '''

    value = []
    description = []
    index = []

    xml_tree = ET.parse(path)  # parse xml from a path
    root = xml_tree.getroot()  # make xml tree structure

    pv_state_shard = root.find('PVStateShard')  # find pv state shard element in root

    for elem in pv_state_shard:  # for each element in pv state shard, find the value for the specified key

        if elem.get('key') == key:

            if len(elem) == 0:  # if the element has only one subelement
                value = elem.get('value')
                break

            else:  # if the element has many subelements (i.e. lots of entries for that key)
                for subelem in elem:
                    value.append(subelem.get('value'))
                    description.append(subelem.get('description'))
                    index.append(subelem.get('index'))
        else:
            for subelem in elem:  # if key not in element, try subelements
                if subelem.get('key') == key:
                    value = elem.get('value')
                    break

        if value:  # if found key in subelement, break the loop
            break

    if not value:  # if no value found at all, raise exception
        raise Exception('ERROR: no element or subelement with that key')

    return value, description, index

pixelSize, _, index = _getPVStateShard('/home/pshah/mnt/qnap/Data/2021-01-11/2021-01-11_s-003/2021-01-11_s-003.xml', 'micronsPerPixel')
for pixelSize, index in zip(pixelSize, index):
    if index == 'XAxis':
        pix_sz_x = float(pixelSize)

print(f'100um in pixels: {int(100 / pix_sz_x)}')







