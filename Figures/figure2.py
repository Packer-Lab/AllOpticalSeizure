import matplotlib.pyplot as plt
import numpy as np
from funcsforprajay.plotting.plotting import plot_bar_with_points

import _utils_.alloptical_plotting as aoplot

# import onePstim superobject that will collect analyses from various individual experiments
from _exp_metainfo_.exp_metainfo import import_resultsobj, ExpMetainfo
from _utils_.io import import_expobj
from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs
from onePexperiment.OnePhotonStimMain import OnePhotonStim

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = import_resultsobj(pkl_path=results_object_path)

# date = '2021-01-24'

# expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial


# %% B) LFP signal with optogenetic stims

date = '2021-01-24'

# pre4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5 / 100 * 100, 3), linewidth=0.5, downsample=True,
                     sz_markings=False, color='black',
                     ylims=[-4, 1], xlims=[105 * expobj.paq_rate, 205 * expobj.paq_rate])

# post4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5 / 100 * 150, 3), linewidth=0.5, downsample=True,
                     sz_markings=False, color='black',
                     ylims=[0, 5], xlims=[10 * expobj.paq_rate, 160 * expobj.paq_rate])

# Avg Flu signal with optogenetic stims

offset = expobj.frame_start_time_actual / expobj.paq_rate

# pre4ap
expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
aoplot.plotMeanRawFluTrace(expobj, stim_span_color='cornflowerblue', x_axis='Time (secs)',
                           xlims=[105 * expobj.fps, 205 * expobj.fps], stim_lines=False,
                           figsize=[5 / 100 * 100, 3])

# post4ap
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
aoplot.plotMeanRawFluTrace(expobj, stim_span_color='cornflowerblue', x_axis='Time (secs)',
                           xlims=[10 * expobj.fps, 160 * expobj.fps], stim_lines=False,
                           figsize=[5 / 100 * 150, 3])

# %% C) avg LFP trace 1p stim plots

date = '2021-01-24'

# pre4ap
pre4ap = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial

assert 'pre' in pre4ap.exptype
aoplot.plot_flu_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue',
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Baseline', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
                                 write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8,
                                 stims_to_analyze=pre4ap.stim_start_frames,
                                 title='Baseline')

# post4ap
post4ap = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

assert 'post' in post4ap.exptype
aoplot.plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue',
                                 stims_to_analyze=post4ap.stims_out_sz,
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Interictal', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
                                 write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8,
                                 stims_to_analyze=post4ap.stims_out_sz,
                                 title='Interictal')

aoplot.plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue',
                                 stims_to_analyze=post4ap.stims_in_sz,
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Ictal', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
                                 write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8,
                                 stims_to_analyze=post4ap.stims_in_sz,
                                 title='Ictal')

# %% D) BAR PLOT OF RESPONSE MAGNITUDE FOR 1P STIM EXPERIMENTS

data = [[rp for rp in onePresults.mean_stim_responses.iloc[:, 1] if rp != '-']]
data.append([rp for rp in onePresults.mean_stim_responses.iloc[:, 2] if rp != '-'])
# data.append([rp for rp in onePresults.mean_stim_responses.iloc[:,3] if rp != '-'])


interictal_response_magnitude = {
    'PS07': [0.2654, 0.1296],
    'PS11': [0.9005, 0.9208],
    'PS18': [1.5309],
    'PS09': [0.2543, 0.6789, 0.763],
    'PS16': [0.1858, 0.2652]
}

baseline_response_magnitude = {
    # 'PS17': [0.1686],  # can't be used for paird pre4ap to interictal plot
    'PS07': [0.3574, 0.3032],
    'PS11': [0.4180, 0.3852],
    'PS18': [0.3594],
    'PS09': [0.2662],
    'PS16': [0.1217, 0.093, 0.087]
}

interictal_response_magnitude_plot = [np.mean(items) for items in interictal_response_magnitude.values()]
baseline_response_magnitude_plot = [np.mean(items) for items in baseline_response_magnitude.values()]

fig, ax = plt.subplots(figsize=[3, 5], dpi = 300)
plot_bar_with_points(data=[baseline_response_magnitude_plot, interictal_response_magnitude_plot],
                     title='response magnitudes', x_tick_labels=['baseline', 'interictal'], paired=True,
                     points=True, bar=False, colors=['gray', 'green'], fig=fig, ax=ax, show=False, s=50,
                     x_label='experiment groups', y_label='Avg. dFF', alpha=1,
                     expand_size_x=0.5, expand_size_y=1.3, shrink_text=1.35, ylims=[0, 2.5])
fig.tight_layout(pad=0.2)
fig.show()
save_path = '/home/pshah/mnt/qnap/Analysis/' + 'onePstim_response_quant'
print('saving fig to: ', save_path)
fig.savefig(fname=save_path + '.png', transparent=True, format='png')
fig.savefig(fname=save_path + '.svg', transparent=True, format='svg')


# todo run paired ttest stats


# %% E) BAR PLOT OF DECAY CONSTANT FOR 1P STIM EXPERIMENTS

data = [list(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -3].notnull()].iloc[:, -3])]
data.append(list(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -2].notnull()].iloc[:, -2]))
# data.append(ls(onePresults.mean_stim_responses[onePresults.mean_stim_responses.iloc[:, -1].notnull()].iloc[:, -1]))

interictal_decay_constant = {
    'PS07': [1.26471, 0.32844],
    'PS11': [1.05109, 0.45986],
    'PS18': [0.65681],
    'PS09': [0.52549, 0.59118, 0.39412],
    'PS16': [0.78811, 0.98517]
}

baseline_decay_constant = {
    # 'PS17': [0.1686],
    'PS07': [0.45972, 0.46595],
    'PS11': [0.45980, 0.52553],
    'PS18': [0.45975],
    'PS09': [0.52546],
    'PS16': [0.45964, 0.39399, 0.46595]
}

baseline_decay_constant_plot = [np.mean(items) for items in baseline_decay_constant.values()]
interictal_decay_constant_plot = [np.mean(items) for items in interictal_decay_constant.values()]


fig, ax = plt.subplots(figsize=[3, 5])
plot_bar_with_points(data=[baseline_decay_constant_plot, interictal_decay_constant_plot], title='decay constants',
                     legend_labels=list(onePresults.mean_stim_responses.columns[-3:]), paired=True, x_tick_labels=['baseline', 'interictal'],
                     points=True, bar=False, colors=['gray', 'green'], fig=fig, ax=ax, show=False,
                     x_label='experiment groups', y_label='Avg. Decay constant (secs.)', alpha=1, s=50,
                     expand_size_x=0.9, expand_size_y=1.2, shrink_text=1.35, ylims=[0.2, 1.1])
fig.tight_layout(pad=0.2)
fig.show()
save_path = '/home/pshah/mnt/qnap/Analysis/' + 'onePstim_decay_quant'
print('saving fig to: ', save_path)
fig.savefig(fname=save_path + '.png', transparent=True, format='png')
fig.savefig(fname=save_path + '.svg', transparent=True, format='svg')


# todo run paired ttest stats




# %% F) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period

# run data analysis
exp_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim()

# make plot
bin_width = int(1 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr / bin_width))) - 1
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width))[:-1] / period

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=600)
for exp, values in exp_sz_occurrence.items():
    plot = values
    ax.bar(theta, plot, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)
ax.set_rmax(1.1)
# ax.set_rticks([1])  # Less radial ticks
ax.set_rlabel_position(-35.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_title("sz probability occurrence (binned every 10s)", va='bottom')
ax.spines['polar'].set_visible(False)
fig.show()
