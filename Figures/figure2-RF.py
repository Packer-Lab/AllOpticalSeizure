# %%
import sys

from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_bar_with_points
from matplotlib.transforms import Bbox
from scipy import stats

from _alloptical_utils import run_for_loop_across_exps
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings, plotLfpSignal, plotMeanRawFluTrace, plot_flu_1pstim_avg_trace, \
    plot_lfp_1pstim_avg_trace
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

import cairosvg
from PIL import Image
from io import BytesIO

from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs, OnePhotonStimResults
from onePexperiment.OnePhotonStimMain import OnePhotonStim, onePresults

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rep_fig_vis as rfv

plot_settings()

Results = OnePhotonStimResults.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

date = '2021-01-24'

save_fig = True

# %% MAKE FIGURE LAYOUT
rfv.set_fontsize(7)

layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.05, 0.80, 0.40, 0.95)},
    'B': {'panel_shape': (2, 2),
          'bound': (0.45, 0.80, 0.95, 0.95)},
    'C': {'panel_shape': (3, 2),
          'bound': (0.05, 0.54, 0.40, 0.72)},
    'D-E': {'panel_shape': (2, 1),
            'bound': (0.51, 0.57, 0.75, 0.72),
            'wspace': 0.9}
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=50)

rfv.naked(axes['A'][0])

# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.



# %% E) BAR PLOT OF RESPONSE DECAY FOR 1P STIM EXPERIMENTS

rfv.add_label_axes(text='E', ax=axes['D-E'][1], y_adjust=0, x_adjust=0.07)

baseline_decay_constant_plot = [np.mean(items) for items in Results.baseline_decay_constant.values()]
interictal_decay_constant_plot = [np.mean(items) for items in Results.interictal_decay_constant.values()]

plot_bar_with_points(data=[baseline_decay_constant_plot, interictal_decay_constant_plot],
                     legend_labels=list(onePresults.mean_stim_responses.columns[-3:]), paired=True, x_tick_labels=['baseline', 'interictal'],
                     points=True, bar=False, colors=['gray', 'green'], fig=fig, ax=axes['D-E'][1], show=False,
                     x_label='', y_label=r'Decay ($\tau$, secs)', alpha=1, s=50, ylims=[0.2, 1.1])

# paired t-test
stats.ttest_rel(baseline_decay_constant_plot, interictal_decay_constant_plot, alternative='two-sided')



# %% D) BAR PLOT OF RESPONSE MAGNITUDE FOR 1P STIM EXPERIMENTS

rfv.add_label_axes(text='D', ax=axes['D-E'][0], y_adjust=0, x_adjust=0.07)

baseline_response_magnitude_plot = [np.mean(items) for items in Results.baseline_response_magnitude.values()][1:]  # excluding first baseline sample, no post4ap for that experiment.
interictal_response_magnitude_plot = [np.mean(items) for items in Results.interictal_response_magnitude.values()]


# fig, ax = plt.subplots(figsize=[3, 5], dpi = 300)
plot_bar_with_points(data=[baseline_response_magnitude_plot, interictal_response_magnitude_plot],
                    x_tick_labels=['baseline', 'interictal'], paired=True,
                     points=True, bar=False, colors=['gray', 'green'], fig=fig, ax=axes['D-E'][0], show=False, s=50,
                     x_label='', y_label='Avg. dFF', alpha=1, ylims=[0, 2.2])


# paired t-test - not significant
stats.ttest_rel(np.array(baseline_response_magnitude_plot), np.array(interictal_response_magnitude_plot), alternative='less')


# t-test
interictal_responses = flattenOnce(Results.interictal_response_magnitude.values())
baseline_responses = flattenOnce(Results.baseline_response_magnitude.values())

stats.ttest_ind(baseline_responses, interictal_responses, alternative='less')

plot_bar_with_points(data=[baseline_responses, interictal_responses],
                    x_tick_labels=['baseline', 'interictal'], paired=False,
                     points=True, bar=False, colors=['gray', 'green'], fig=fig, ax=axes['D-E'][0], show=False, s=50,
                     x_label='', y_label='Avg. dFF', alpha=1, ylims=[0, 2.2])


# wilcoxon

stats.wilcoxon(np.array(baseline_response_magnitude_plot) - np.array(interictal_response_magnitude_plot))



# %% B)

rfv.add_label_axes(text='B', ax=axes['B'][0, 0], y_adjust=0)


# pre4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=0.5, downsample=True,
              sz_markings=False, color='black', fig=fig, ax=axes['B'][0, 0], show=False, title='',
              ylims=[-4, 1], xlims=[105, 205])
axes['B'][0, 0].set_title('')
axes['B'][0, 0].axis('off')

# pre4ap
expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotMeanRawFluTrace(expobj, stim_span_color='cornflowerblue', x_axis='Time (secs)',
                    xlims=[105 * expobj.fps, 205 * expobj.fps], stim_lines=False, fig=fig, ax=axes['B'][0, 1],
                    show=False)
axes['B'][0, 1].set_title('')
axes['B'][0, 1].axis('off')



# %%


# post4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=0.5, downsample=True, sz_markings=False, color='black', fig=fig, ax=axes['B'][1, 0],
              show=False, ylims=[0, 5], xlims=[10, 160])
axes['B'][1, 0].set_title('')
axes['B'][1, 0].axis('off')
rfv.add_scale_bar(ax=axes['B'][1, 0], length=(1, 10), bartype='L', text = ('1 mV', '10 s'), loc=(170, 0), text_offset=[5, 0.6], fs=5)


# Avg Flu signal with optogenetic stims
offset = expobj.frame_start_time_actual / expobj.paq_rate

# post4ap
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotMeanRawFluTrace(expobj, stim_span_color='cornflowerblue', x_axis='Time (secs)', xlims=[10 * expobj.fps, 160 * expobj.fps],
                    stim_lines=False, fig=fig, ax=axes['B'][1, 1], show=False)
axes['B'][1, 1].set_title('')
axes['B'][1, 1].axis('off')
rfv.add_scale_bar(ax=axes['B'][1, 1], length=(500, 10 * expobj.fps), bartype='L', text = ('500 a.u.', '10 s'), loc=(170 * expobj.fps, 0),
                  text_offset=[5 * expobj.fps, 250], fs=5)



# fig.show()

# %% C) avg LFP trace 1p stim plots

rfv.add_label_axes(text='C', ax=axes['C'][0,0], y_adjust=0)

# pre4ap
pre4ap = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial

assert 'pre' in pre4ap.exptype
fig, ax = plot_lfp_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, fig=fig, ax=axes['C'][0, 0], show=False,
                                    write_full_text=False, optoloopback=True, stims_to_analyze=pre4ap.stim_start_frames, title='Baseline')
ax.axis('off')
# fig.show()


fig, ax = plot_flu_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', fig=fig,
                                    ax=axes['C'][0, 1], show=False, y_axis='dff', quantify=False, title='Baseline', ylims=[-0.5, 2.0])
ax.axis('off')


# post4ap
post4ap = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

assert 'post' in post4ap.exptype
fig, ax = plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', fig=fig, ax=axes['C'][1, 1], show=False,
                          stims_to_analyze=post4ap.stims_out_sz, y_axis='dff', quantify=False, title='Interictal',
                          ylims=[-0.5, 2.0])
ax.axis('off')


fig, ax = plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, fig=fig, ax=axes['C'][1, 0], show=False,
                          write_full_text=False, optoloopback=True, stims_to_analyze=post4ap.stims_out_sz,
                          title='Interictal')
ax.axis('off')

fig, ax = plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', fig=fig, ax=axes['C'][2, 1], show=False,
                          stims_to_analyze=post4ap.stims_in_sz, y_axis='dff', quantify=False, title='Ictal',
                          ylims=[-0.5, 2.0])
ax.axis('off')
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(0.5, 1), bartype='L', text = ('0.5 dFF', '1 s'), loc=(x - 1, y - 0.5), text_offset=[0.50, 0.25], fs=5)
# fig.show()

fig, ax = plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, fig=fig, ax=axes['C'][2,0], show=False,
                          write_full_text=False, optoloopback=True, stims_to_analyze=post4ap.stims_in_sz, title='Ictal')
ax.axis('off')
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(1, 0.25), bartype='L', text = ('1 mV', '0.25 s'), loc=(x-0.25, y-1), text_offset=[0.07, 0.50], fs = 5)
#
# fig.show()



# %% F) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period

# run data analysis
exp_sz_occurrence, total_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim(Results=Results,
                                                                                                    rerun=0)

expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

# make plot
bin_width = int(1 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr // bin_width)))
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr // bin_width)) / period

bbox = Bbox.from_extents(0.80, 0.58, 0.95, 0.72)
_axes = np.empty(shape=(1, 1), dtype=object)
ax = fig.add_subplot(projection = 'polar')
ax.set_position(pos=bbox)
rfv.add_label_axes(text='F', ax=ax, y_adjust=0.01)

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300, figsize=(3, 3))

# by experiment
# for exp, values in exp_sz_occurrence.items():
#     plot = values
#     ax.bar(theta, plot, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)

# across all seizures
total_sz = np.sum(np.sum(total_sz_occurrence, axis=0))
sz_prob = np.sum(total_sz_occurrence, axis=0) / total_sz

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color='cornflowerblue', lw=0.3)

ax.set_rmax(1.1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # radial ticks
ax.set_rlabel_position(-120)
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels(['0', '', '50', ''])
# ax.set_title("sz probability occurrence (binned every 1s)", va='bottom')
ax.spines['polar'].set_visible(False)

fig.show()


# %%
if save_fig:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure2-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure2-RF.svg")


