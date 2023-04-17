# %%
import sys

from funcsforprajay.plotting.plotting import plot_bar_with_points
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from scipy import stats

from _exp_metainfo_.exp_metainfo import ExpMetainfo, baseline_color, interictal_color, fontsize_extraplot
from _utils_.alloptical_plotting import plot_settings, plotLfpSignal, plotMeanRawFluTrace, plot_flu_1pstim_avg_trace, \
    plot_lfp_1pstim_avg_trace, save_figure
from _utils_.io import import_expobj

from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs, OnePhotonStimResults
from onePexperiment.OnePhotonStimMain import OnePhotonStim, onePresults

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import rep_fig_vis as rfv

from pycircstat.tests import vtest

Results: OnePhotonStimResults = OnePhotonStimResults.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

date = '2021-01-24'

save_fig = True

stim_color = ExpMetainfo.figures.colors['1p stim span']
# stim_color = 'powderblue'

# %% MAKE FIGURE LAYOUT
fs = fontsize_extraplot
rfv.set_fontsize(fontsize_extraplot)


layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.05, 0.80, 0.40, 0.95)},
    'B': {'panel_shape': (2, 2),
          'bound': (0.43, 0.80, 0.93, 0.95),
          'hspace': 0.2},
    'D': {'panel_shape': (2, 2),
          'bound': (0.33, 0.58, 0.55, 0.75),
          'hspace': 0.1},
    'E-F': {'panel_shape': (2, 1),
            'bound': (0.66, 0.62, 0.91, 0.75),
            'wspace': 1.4}
}

test = 0
save_fig = True if not test > 0 else False
dpi = 100 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.

rfv.naked(axes['A'][0])
rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0)
rfv.add_label_axes(text='E', ax=axes['E-F'][0], y_adjust=0.01, x_adjust=0.08)
rfv.add_label_axes(text='F', ax=axes['E-F'][1], y_adjust=0.01, x_adjust=0.1)
rfv.add_label_axes(text='B', ax=axes['B'][0, 0], y_adjust=0)

print('\n\n')


# %% B) representative plot of onePhoton experiment


# pre4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=ExpMetainfo.figure_settings['lfp - lw'], downsample=True, stim_span_color=stim_color,
              sz_markings=False, color='black', fig=fig, ax=axes['B'][0, 0], show=False, title='',
              ylims=[-4, 1], xlims=[105, 205])
axes['B'][0, 0].set_title('')
axes['B'][0, 0].axis('off')

# pre4ap
expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotMeanRawFluTrace(expobj, stim_span_color=stim_color, x_axis='Time (secs)', linewidth = ExpMetainfo.figure_settings["gcamp - FOV - lw"],
                    xlims=[105 * expobj.fps, 205 * expobj.fps], stim_lines=False, fig=fig, ax=axes['B'][0, 1],
                    show=False)
axes['B'][0, 1].set_title('')
axes['B'][0, 1].axis('off')

# post4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=ExpMetainfo.figure_settings['lfp - lw'], downsample=True, sz_markings=False, color='black', fig=fig, stim_span_color=stim_color,
              ax=axes['B'][1, 0], show=False, ylims=[0, 5], xlims=[10, 160])
axes['B'][1, 0].set_title('')
axes['B'][1, 0].axis('off')
rfv.add_scale_bar(ax=axes['B'][1, 0], length=(1, 10), bartype='L', text=('1\nmV', '10 s'), loc=(180, 0),
                  text_offset=[2, 0.8], fs=ExpMetainfo.figure_settings["fontsize - intraplot"])

# Avg Flu signal with optogenetic stims
offset = expobj.frame_start_time_actual / expobj.paq_rate

# post4ap
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotMeanRawFluTrace(expobj, stim_span_color=stim_color, x_axis='Time (secs)',
                    xlims=[10 * expobj.fps, 160 * expobj.fps], linewidth=ExpMetainfo.figure_settings["gcamp - FOV - lw"],
                    stim_lines=False, fig=fig, ax=axes['B'][1, 1], show=False)
axes['B'][1, 1].set_title('')
axes['B'][1, 1].axis('off')
rfv.add_scale_bar(ax=axes['B'][1, 1], length=(500, 10 * expobj.fps), bartype='L', text=('500\na.u.', '10 s'),
                  loc=(180 * expobj.fps, 0), text_offset=[2 * expobj.fps, 440], fs=ExpMetainfo.figure_settings["fontsize - intraplot"])



# %% D) avg LFP trace 1p stim plots

rfv.add_label_axes(text='D', ax=axes['D'][0, 0], y_adjust=0.01)

# pre4ap
pre4ap = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial

assert 'pre' in pre4ap.exptype
fig, ax = plot_lfp_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, pre_stim=0.15, post_stim=0.85,
                                    fig=fig, ax=axes['D'][0, 0], show=False, write_full_text=False, optoloopback=True, stims_to_analyze=pre4ap.stim_start_frames,
                                    title='Baseline', fillcolor=ExpMetainfo.figure_settings['colors']['baseline'], spancolor=stim_color)
ax.axis('off')
ax.text(s='LFP', x=-0.17, y=-1.65, ha='center', rotation=90, fontsize=8)
ax.set_title('Baseline', fontsize=ExpMetainfo.figure_settings["fontsize - title"])

fig, ax = plot_flu_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, stim_span_color=stim_color, fig=fig,
                                    ax=axes['D'][0, 1], show=False, y_axis='dff', quantify=False, title='Baseline', pre_stim=0.85, post_stim=3.60,
                                    ylims=[-0.5, 2.0], fillcolor=ExpMetainfo.figure_settings['colors']['baseline'])
ax.axis('off')
ax.text(s=r'FOV Ca$^{2+}$', x=-0.9, y=0.25, ha='center', rotation=90, fontsize=8)

# post4ap
post4ap = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

assert 'post' in post4ap.exptype
fig, ax = plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color=stim_color, fig=fig, pre_stim=0.85, post_stim=3.60,
                                    ax=axes['D'][1, 1], show=False, stims_to_analyze=post4ap.stims_out_sz, y_axis='dff', quantify=False,
                                    title='Interictal', ylims=[-0.5, 2.0], fillcolor=ExpMetainfo.figure_settings['colors']['interictal'])
ax.axis('off')
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(0.5, 1), bartype='L', text=(f'0.5\ndFF', '1 s'), loc=(x - 1, y - 0.9),
                  text_offset=[0.10, 0.35], fs=ExpMetainfo.figures.fontsize['intraplot'])



fig, ax = plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.15, post_stim=0.85,
                                    fig=fig, ax=axes['D'][1, 0], show=False, write_full_text=False, optoloopback=True, stims_to_analyze=post4ap.stims_out_sz,
                                    title='Interictal', fillcolor=ExpMetainfo.figure_settings['colors']['interictal'], spancolor=stim_color)
ax.axis('off')
ax.set_title('Interictal', fontsize=ExpMetainfo.figure_settings["fontsize - title"])
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(1, 0.25), bartype='L', text=('1\nmV', '0.25 s'), loc=(x - 0.25, y - 1.2),
                  text_offset=[0.035, 0.7], fs=ExpMetainfo.figures.fontsize['intraplot'])

# fig.show()

# fig, ax = plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', fig=fig,
#                                     ax=axes['D'][2, 1], show=False, stims_to_analyze=post4ap.stims_in_sz, y_axis='dff', quantify=False, title='Ictal',
#                                     ylims=[-0.5, 2.0])
# ax.axis('off')

# fig, ax = plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75,
#                                     fig=fig, ax=axes['D'][2, 0], show=False,
#                                     write_full_text=False, optoloopback=True, stims_to_analyze=post4ap.stims_in_sz,
#                                     title='Ictal')
# ax.axis('off')
# ax.set_title('Ictal', fontsize=10, fontweight='semibold')





# %% E) BAR PLOT OF RESPONSE MAGNITUDE FOR 1P STIM EXPERIMENTS - BY INDIVIDUAL STIMS
ax=axes['E-F'][0]

# individual trials photostim responses
baseline_response_magnitudes = Results.photostim_responses['baseline']
interictal_response_magnitudes = Results.photostim_responses['interictal']
interictal_response_magnitudes_szexclude = Results.photostim_responses['interictal - sz excluded']
interictal_response_magnitudes_presz = Results.photostim_responses['interictal - presz']
interictal_response_magnitudes_postsz = Results.photostim_responses['interictal - postsz']

baseline_resposnes = []
interictal_resposnes = []
for trial, responses in baseline_response_magnitudes.items():
    baseline_resposnes.extend(list(responses))
for trial, responses in interictal_response_magnitudes.items():
    interictal_resposnes.extend(list(responses))
interictal_resposnes_szexclude = []
for trial, responses in interictal_response_magnitudes_szexclude.items():
    interictal_resposnes_szexclude.extend(list(responses))

# import seaborn as sns
# sns.violinplot(data=[baseline_resposnes, interictal_resposnes], ax=axes['E-F'][0])

# experimental average photostim responses
baseline_response_magnitudes_exp = [np.mean(x) for x in list(Results.baseline_response_magnitude.values())]
interictal_response_magnitudes_exp = [np.mean(x) for x in list(Results.interictal_response_magnitude.values())]

# STATS
# t-test - individual sessions
print(f"P(t-test - (indiv. trials) response: baseline ({len(baseline_resposnes)} trials) vs. interictal ({len(interictal_resposnes_szexclude)} trials)): \n\t\t{stats.ttest_ind(baseline_resposnes, interictal_resposnes_szexclude)[1]:.3e}")

# print mean and stdev of response magnitudes
print(f"Mean response magnitude (baseline): {np.mean(baseline_resposnes):.3f} +/- {np.std(baseline_resposnes):.3f}")
print(f"Mean response magnitude (interictal): {np.mean(interictal_resposnes_szexclude):.3f} +/- {np.std(interictal_resposnes_szexclude):.3f}")


# VIOLIN PLOT
# vp = ax.violinplot([baseline_resposnes, interictal_resposnes_szexclude], showmeans=True, showextrema=False, showmedians=False,
#                    widths=0.7)
# ax.set_xlim([0.2, 2.8])
# ax.set_ylim([0, 1.25])
# ax.set_xticks([1,2], ['Baseline', 'Interictal'], fontsize=ExpMetainfo.figure_settings["fontsize - extraplot"], rotation=45)
# # Set the color of the boxes to blue and orange
# vp['bodies'][0].set(facecolor=ExpMetainfo.figure_settings['colors']['baseline'], edgecolor=ExpMetainfo.figure_settings['colors']['baseline'])
# vp['bodies'][1].set(facecolor=ExpMetainfo.figure_settings['colors']['interictal'], edgecolor=ExpMetainfo.figure_settings['colors']['interictal'])
# ax.set_ylabel('Avg. dFF', fontsize=ExpMetainfo.figure_settings["fontsize - extraplot"])
# Set the widths of the violins to 0.2
# for body in vp['bodies']:
#     body.set_widths(0.2)

# fig, ax = plt.subplots(figsize=[2, 3], dpi = 100)
plot_bar_with_points(data=[baseline_resposnes, interictal_resposnes_szexclude],
                     x_tick_labels=['Baseline', 'Interictal'], fontsize=fontsize_extraplot,
                     points=False, bar=True, colors=[baseline_color, interictal_color], fig=fig, ax=ax, show=False, s=10,
                     x_label='', y_label='Avg. dFF', alpha=0.7, lw=0.75, ylims=[0, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# fig.tight_layout(pad=0.2)




# %% F) BAR PLOT OF RESPONSE DECAY FOR 1P STIM EXPERIMENTS - changing to individual stims - '22 dec 19
ax = axes['E-F'][1]

# individual trials photostim decays
baseline_decay_magnitudes = Results.decay_constants['baseline']
interictal_decay_magnitudes_szexclude = Results.decay_constants['interictal - sz excluded']

baseline_decays = []
interictal_decays_szexclude = []
for trial, decays in baseline_decay_magnitudes.items():
    decays = [decay for decay in decays if (decay > 0.0 and decay is not None)]
    baseline_decays.extend(list(decays))
for trial, decays in interictal_decay_magnitudes_szexclude.items():
    decays = [decay for decay in decays if (decay > 0.0 and decay is not None)]
    interictal_decays_szexclude.extend(list(decays))


# STATS
# t-test - individual sessions
print(f"t-test - (indiv. trials) decay constants: baseline ({len(baseline_decays)} trials) vs. interictal ({len(interictal_decays_szexclude)} trials): \n\t\t p = {stats.ttest_ind(baseline_decays, interictal_decays_szexclude)[1]:.3e}")

# print mean and stdev of decay constants
print(f"Mean decay constant (baseline): {np.mean(baseline_decays):.3f} +/- {np.std(baseline_decays):.3f}")
print(f"Mean decay constant (interictal): {np.mean(interictal_decays_szexclude):.3f} +/- {np.std(interictal_decays_szexclude):.3f}")


# VIOLIN PLOT
# # add violin plot of baseline vs. interictal decays, with color of baseline as royalblue and interictal as darkorange to ax
# vp = ax.violinplot([baseline_decays, interictal_decays_szexclude], showmeans=True, showextrema=False, showmedians=False,
#                    widths=0.7)
# ax.set_xlim([0.2, 2.8])
# ax.set_ylim([0, 1.0])
# ax.set_xticks([1,2], ['Baseline', 'Interictal'], fontsize=ExpMetainfo.figure_settings["fontsize - extraplot"], rotation=45)
#
# # Set the color of the boxes to blue and orange
# vp['bodies'][0].set(facecolor=ExpMetainfo.figure_settings['colors']['baseline'], edgecolor=ExpMetainfo.figure_settings['colors']['baseline'])
# vp['bodies'][1].set(facecolor=ExpMetainfo.figure_settings['colors']['interictal'], edgecolor=ExpMetainfo.figure_settings['colors']['interictal'])
# ax.set_ylabel(r'Decay ($\tau$, secs)', fontsize=ExpMetainfo.figure_settings["fontsize - extraplot"])

# fig.show()

# fig, ax = plt.subplots(figsize=[2, 3], dpi = 100)
plot_bar_with_points(data=[baseline_decays, interictal_decays_szexclude],
                     x_tick_labels=['Baseline', 'Interictal'], fontsize=fontsize_extraplot,
                     points=False, bar=True, colors=[ExpMetainfo.figure_settings['colors']['baseline'], ExpMetainfo.figure_settings['colors']['interictal']], fig=fig, ax=ax, show=False, s=10,
                     x_label='', y_label=r'Decay ($\tau$, secs)', alpha=1, lw=0.75, ylims=[0, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# fig.show()
# fig.tight_layout(pad=0.2)





# ### archiving below '22 dec 19
# ax=axes['E-F'][1]
# baseline_decay_constant_plot = [np.mean(items) for items in Results.baseline_decay_constant.values()]
# interictal_decay_constant_plot = [np.mean(items) for items in Results.interictal_decay_constant.values()]
#
# # STATS
# print(
#     f"P(paired t-test - decay - baseline vs. interictal): {stats.ttest_rel(baseline_decay_constant_plot, interictal_decay_constant_plot)[1]:.3f}")
# print(
#     f"P(t-test - decay - baseline vs. interictal): {stats.ttest_ind(baseline_decay_constant_plot, interictal_decay_constant_plot)[1]:.3f}")
#
# # make plot
# plot_bar_with_points(data=[baseline_decay_constant_plot, interictal_decay_constant_plot],
#                      legend_labels=list(onePresults.mean_stim_responses.columns[-3:]), paired=True,
#                      x_tick_labels=['Baseline', 'Interictal'], fs=ExpMetainfo.figure_settings["fontsize - extraplot"],
#                      points=True, bar=False, colors=['royalblue', 'forestgreen'], fig=fig, ax=ax, show=False,
#                      x_label='', y_label=r'Decay ($\tau$, secs)', alpha=1, s=35, ylims=[0.3, 1.1], fontsize=ExpMetainfo.figure_settings["fontsize - extraplot"])
#
# print('\n\n')




# %% C) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period

# run data analysis
exp_sz_occurrence, total_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim(Results=Results,
                                                                                                    rerun=0)

expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

# make plot
bin_width = int(1 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr // bin_width)))
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr // bin_width)) / period

# bbox = Bbox.from_extents(0.70, 0.60, 0.84, 0.74)
bbox = Bbox.from_extents(0.0, 0.60, 0.24, 0.74)
# _axes = np.empty(shape=(1, 1), dtype=object)
ax = fig.add_subplot(projection='polar')

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300, figsize=(3, 3))


# by experiment
# for exp, values in exp_sz_occurrence.items():
#     plot = values
#     ax.bar(theta, plot, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)

# across all seizures
total_sz = np.sum(np.sum(total_sz_occurrence, axis=0))
sz_prob = np.sum(total_sz_occurrence, axis=0) / total_sz

pval, z = vtest(alpha=theta, mu=0, w=np.sum(total_sz_occurrence, axis=0))
print(f'pval for oneP stim seizure incidence is: {pval}')


ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color=ExpMetainfo.figures.colors['general'], lw=0.3, edgecolor='black')

ax.set_rmax(1.1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # radial ticks
ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=ExpMetainfo.figure_settings['fontsize - intraplot'])  # radial ticks
ax.set_rlabel_position(-60)
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels([r'$\it{\Theta}$ = 0', '', '', ''], fontsize=ExpMetainfo.figure_settings['fontsize - extraplot'])
# ax.set_title("sz probability occurrence (binned every 1s)", va='bottom')
ax.spines['polar'].set_visible(False)
ax.set_title(label='Seizure probability', fontsize=10, va='bottom')
ax.set_position(pos=bbox)
rfv.add_label_axes(text='C', ax=ax, y_adjust=0.02, x_adjust=0.025)




# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure2-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure2-RF.pdf")

fig.show()

