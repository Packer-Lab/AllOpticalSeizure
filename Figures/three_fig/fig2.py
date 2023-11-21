# %% FIGURE 2 - alloptical interrogation of

"""
Figure 2: Widescale and single-neuronal excitability after focal 4-AP injection
"""

import sys

import pandas as pd
import numpy as np
from funcsforprajay.funcs import flattenOnce
from matplotlib import image as mpimg
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
fig_title = f'fig2_widefieldstim_alloptical'
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/3fig/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

import rep_fig_vis as rfv
from _utils_.rfv_funcs import add_scale_bar
from matplotlib.transforms import Bbox
from funcsforprajay.plotting.plotting import plot_bar_with_points

from _exp_metainfo_.exp_metainfo import ExpMetainfo, baseline_color, interictal_color
from _utils_.alloptical_plotting import plotLfpSignal, plotMeanRawFluTrace, plot_flu_1pstim_avg_trace, \
    plot_lfp_1pstim_avg_trace
from _utils_.io import import_expobj
from _utils_.alloptical_plotting import save_figure

from onePexperiment.OnePhotonStimAnalysis_main import OnePhotonStimAnalysisFuncs, OnePhotonStimResults
from onePexperiment.OnePhotonStimMain import OnePhotonStim

from pycircstat.tests import vtest

Results: OnePhotonStimResults = OnePhotonStimResults.load()

date = '2021-01-24'

save_fig = True

stim_color = ExpMetainfo.figures.colors['1p stim span']
# stim_color = 'powderblue'

# %% MAKE FIGURE LAYOUT

fs_intra = ExpMetainfo.figure_settings['fontsize - intraplot']
fs_extra = ExpMetainfo.figure_settings['fontsize - extraplot']

layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.03, 0.80, 0.40, 0.95)},
    'B': {'panel_shape': (2, 2),
          'bound': (0.43, 0.80, 0.93, 0.95),
          'hspace': 0.2},
    'D': {'panel_shape': (2, 2),
          'bound': (0.30, 0.58, 0.56, 0.75),
          'hspace': 0.1},
    'E-F': {'panel_shape': (2, 1),
            'bound': (0.68, 0.635, 0.95, 0.75),
            'wspace': 1.4},
    'G': {'panel_shape': (1, 1),
          'bound': (0.03, 0.23, 0.26, 0.57)},
    'Htop': {'panel_shape': (2, 1),
             'bound': (0.30, 0.45, 0.94, 0.54)},
    'Hbottom': {'panel_shape': (2, 1),
                'bound': (0.30, 0.23, 0.94, 0.44)},
    'J': {'panel_shape': (2, 1),
          'bound': (0.30, 0.07, 0.56, 0.20),
          'wspace': 1.0},
    'K': {'panel_shape': (1, 1),
          'bound': (0.68, 0.07, 0.76, 0.20)},
    'L': {'panel_shape': (1, 1),
          'bound': (0.87, 0.07, 0.95, 0.20)
          }
}

test = 0
save_fig = True if not test > 0 else False
dpi = 100 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes,
                            show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.

rfv.naked(axes['A'][0])
# rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0)
# rfv.add_label_axes(text='E', ax=axes['E-F'][0], y_adjust=0.01, x_adjust=0.08)
# rfv.add_label_axes(text='F', ax=axes['E-F'][1], y_adjust=0.01, x_adjust=0.1)
# rfv.add_label_axes(text='B', ax=axes['B'][0, 0], y_adjust=0)

print('\n\n')

# %% WIDEFIELD STIMULATION MEASUREMENT OF EXCITABILITY

print('\n\n')

################################################
# D) avg LFP trace 1p stim plots ################################################
################################################

pre4ap = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
post4ap = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

##
ax = axes['D'][0, 0]
fig, ax = plot_lfp_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, pre_stim=0.15, post_stim=0.85,
                                    fig=fig, ax=ax, show=False, write_full_text=False, optoloopback=True,
                                    stims_to_analyze=pre4ap.stim_start_frames,
                                    title='Baseline', fillcolor=ExpMetainfo.figure_settings['colors']['baseline'],
                                    spancolor=stim_color)
ax.axis('off')
ax.text(s='LFP', x=-0.17, y=-1.65, ha='center', rotation=90, fontsize=fs_extra)
ax.set_title('Baseline', fontsize=fs_extra)

##
ax = axes['D'][1, 0]
fig, ax = plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.15, post_stim=0.85,
                                    fig=fig, ax=ax, show=False, write_full_text=False, optoloopback=True,
                                    stims_to_analyze=post4ap.stims_out_sz,
                                    title='Interictal', fillcolor=ExpMetainfo.figure_settings['colors']['interictal'],
                                    spancolor=stim_color)
ax.axis('off')
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(1, 0.25), bartype='L', text=('1\nmV', '0.25 s'), loc=(x - 0.25, y - 1.2),
                  text_offset=[0.035, 0.7], fs=fs_intra)
ax.set_title('Interictal', fontsize=fs_extra)

##
ax = axes['D'][0, 1]
fig, ax = plot_flu_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, stim_span_color=stim_color, fig=fig,
                                    ax=ax, show=False, y_axis='dff', quantify=False, title='Baseline', pre_stim=0.85,
                                    post_stim=3.60, ylims=[-0.5, 2.0],
                                    fillcolor=ExpMetainfo.figure_settings['colors']['baseline'])
ax.axis('off')
ax.text(s=r'FOV Ca$^{2+}$', x=-0.9, y=0.25, ha='center', rotation=90, fontsize=fs_extra)

##
# post4ap
ax = axes['D'][1, 1]
fig, ax = plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color=stim_color,
                                    fig=fig, pre_stim=0.85,
                                    post_stim=3.60, ax=ax, show=False, stims_to_analyze=post4ap.stims_out_sz,
                                    y_axis='dff', quantify=False,
                                    title='Interictal', ylims=[-0.5, 2.0],
                                    fillcolor=ExpMetainfo.figure_settings['colors']['interictal'])
ax.axis('off')
x = ax.get_xlim()[1]
y = ax.get_ylim()[1]
rfv.add_scale_bar(ax=ax, length=(0.5, 1), bartype='L', text=(f'0.5\ndFF', '1 s'), loc=(x - 1, y - 0.9),
                  text_offset=[0.10, 0.35], fs=fs_intra)

################################################
# B) representative plot of onePhoton experiment ################################################
################################################

# pre4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=ExpMetainfo.figure_settings['lfp - lw'], downsample=True,
              stim_span_color=stim_color,
              sz_markings=False, color='black', fig=fig, ax=axes['B'][0, 0], show=False, title='',
              ylims=[-4, 1], xlims=[105, 205])
axes['B'][0, 0].set_title('')
axes['B'][0, 0].axis('off')

# pre4ap
expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
plotMeanRawFluTrace(expobj, stim_span_color=stim_color, x_axis='Time (secs)',
                    linewidth=ExpMetainfo.figure_settings["gcamp - FOV - lw"],
                    xlims=[105 * expobj.fps, 205 * expobj.fps], stim_lines=False, fig=fig, ax=axes['B'][0, 1],
                    show=False)
axes['B'][0, 1].set_title('')
axes['B'][0, 1].axis('off')

# post4ap
expobj: OnePhotonStim = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotLfpSignal(expobj, x_axis='time', linewidth=ExpMetainfo.figure_settings['lfp - lw'], downsample=True,
              sz_markings=False, color='black', fig=fig, stim_span_color=stim_color,
              ax=axes['B'][1, 0], show=False, ylims=[0, 5], xlims=[10, 160])
axes['B'][1, 0].set_title('')
axes['B'][1, 0].axis('off')
add_scale_bar(ax=axes['B'][1, 0], length=(1, 10), bartype='L', text=('1 mV', '10 s'), loc=(180, 0),
              text_offset=[10, 1], fontsize=fs_intra)

# Avg Flu signal with optogenetic stims
offset = expobj.frame_start_time_actual / expobj.paq_rate

# post4ap
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
plotMeanRawFluTrace(expobj, stim_span_color=stim_color, x_axis='Time (secs)',
                    xlims=[10 * expobj.fps, 160 * expobj.fps],
                    linewidth=ExpMetainfo.figure_settings["gcamp - FOV - lw"],
                    stim_lines=False, fig=fig, ax=axes['B'][1, 1], show=False)
axes['B'][1, 1].set_title('')
axes['B'][1, 1].axis('off')
add_scale_bar(ax=axes['B'][1, 1], length=(500, 10 * expobj.fps), bartype='L', text=('500 a.u.', '10 s'),
              loc=(180 * expobj.fps, 0), text_offset=[10 * expobj.fps, 420], fontsize=fs_intra)

################################################
# E) BAR PLOT OF RESPONSE MAGNITUDE FOR 1P STIM EXPERIMENTS - BY INDIVIDUAL STIMS ################################################
################################################

ax = axes['E-F'][0]

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
print(
    f"P(t-test - (indiv. trials) response: baseline ({len(baseline_resposnes)} trials) vs. interictal ({len(interictal_resposnes_szexclude)} trials)): \n\t\t{stats.ttest_ind(baseline_resposnes, interictal_resposnes_szexclude)[1]:.3e}")

# print mean and stdev of response magnitudes
print(f"Mean response magnitude (baseline): {np.mean(baseline_resposnes):.3f} +/- {np.std(baseline_resposnes):.3f}")
print(
    f"Mean response magnitude (interictal): {np.mean(interictal_resposnes_szexclude):.3f} +/- {np.std(interictal_resposnes_szexclude):.3f}")

# VIOLIN PLOT
# vp = ax.violinplot([baseline_resposnes, interictal_resposnes_szexclude], showmeans=True, showextrema=False, showmedians=False,
#                    widths=0.7)
# ax.set_xlim([0.2, 2.8])
# ax.set_ylim([0, 1.25])
# ax.set_xticks([1,2], ['Baseline', 'Interictal'], fs=fs_extra, rotation=45)
# # Set the color of the boxes to blue and orange
# vp['bodies'][0].set(facecolor=ExpMetainfo.figure_settings['colors']['baseline'], edgecolor=ExpMetainfo.figure_settings['colors']['baseline'])
# vp['bodies'][1].set(facecolor=ExpMetainfo.figure_settings['colors']['interictal'], edgecolor=ExpMetainfo.figure_settings['colors']['interictal'])
# ax.set_ylabel('Avg. dFF', fs=fs_extra)
# Set the widths of the violins to 0.2
# for body in vp['bodies']:
#     body.set_widths(0.2)

# fig, ax = plt.subplots(figsize=[2, 3], dpi = 100)
plot_bar_with_points(data=[baseline_resposnes, interictal_resposnes_szexclude],
                     x_tick_labels=['Baseline', 'Interictal'], fontsize=fs_extra,
                     points=False, bar=True, colors=[baseline_color, interictal_color], fig=fig, ax=ax, show=False,
                     s=10, x_label='', y_label='Avg. dFF', alpha=0.7, lw=0.75, ylims=[0, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# fig.tight_layout(pad=0.2)


################################################
# F) BAR PLOT OF RESPONSE DECAY FOR 1P STIM EXPERIMENTS - changing to individual stims - '22 dec 19 ################################################
################################################
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
print(
    f"t-test - (indiv. trials) decay constants: baseline ({len(baseline_decays)} trials) vs. interictal ({len(interictal_decays_szexclude)} trials): \n\t\t p = {stats.ttest_ind(baseline_decays, interictal_decays_szexclude)[1]:.3e}")

# print mean and stdev of decay constants
print(f"Mean decay constant (baseline): {np.mean(baseline_decays):.3f} +/- {np.std(baseline_decays):.3f}")
print(
    f"Mean decay constant (interictal): {np.mean(interictal_decays_szexclude):.3f} +/- {np.std(interictal_decays_szexclude):.3f}")

# VIOLIN PLOT
# # add violin plot of baseline vs. interictal decays, with color of baseline as royalblue and interictal as darkorange to ax
# vp = ax.violinplot([baseline_decays, interictal_decays_szexclude], showmeans=True, showextrema=False, showmedians=False,
#                    widths=0.7)
# ax.set_xlim([0.2, 2.8])
# ax.set_ylim([0, 1.0])
# ax.set_xticks([1,2], ['Baseline', 'Interictal'], fs=fs_extra, rotation=45)
#
# # Set the color of the boxes to blue and orange
# vp['bodies'][0].set(facecolor=ExpMetainfo.figure_settings['colors']['baseline'], edgecolor=ExpMetainfo.figure_settings['colors']['baseline'])
# vp['bodies'][1].set(facecolor=ExpMetainfo.figure_settings['colors']['interictal'], edgecolor=ExpMetainfo.figure_settings['colors']['interictal'])
# ax.set_ylabel(r'Decay ($\tau$, secs)', fs=fs_extra)

# fig.show()

# fig, ax = plt.subplots(figsize=[2, 3], dpi = 100)
plot_bar_with_points(data=[baseline_decays, interictal_decays_szexclude],
                     x_tick_labels=['Baseline', 'Interictal'], fontsize=fs_extra,
                     points=False, bar=True, colors=[ExpMetainfo.figure_settings['colors']['baseline'],
                                                     ExpMetainfo.figure_settings['colors']['interictal']], fig=fig,
                     ax=ax, show=False, s=10,
                     x_label='', y_label=r'Decay ($\tau$, secs)', alpha=1, lw=0.75, ylims=[0, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

################################################
# C) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period ################################################
################################################

# run data analysis
exp_sz_occurrence, total_sz_occurrence = OnePhotonStimAnalysisFuncs.collectSzOccurrenceRelativeStim(Results=Results,
                                                                                                    rerun=0)
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

# make plot
bin_width = int(1 * expobj.fps)
period = len(np.arange(0, (expobj.stim_interval_fr // bin_width)))
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr // bin_width)) / period

# bbox = Bbox.from_extents(0.0, 0.60, 0.24, 0.74)
bbox = Bbox.from_extents(0.02, 0.63, 0.20, 0.71)
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

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color=ExpMetainfo.figures.colors['general'],
       lw=0.7, edgecolor='black')

ax.set_rmax(1.1)
ax.set_rlabel_position(-76)
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels([r'$\it{\Theta}$ = 0', '', '', ''], fontsize=fs_extra)
# ax.set_title("sz probability occurrence (binned every 1s)", va='bottom')
ax.spines['polar'].set_visible(False)
ax.set_title(label='Seizure probability\n(widefield stim.)', fontsize=fs_extra, va='bottom')
ax.set_position(pos=bbox)
ax.set_rticks([0.5, 1.0])  # radial ticks
ax.set_yticklabels(['0.5', '1.0'], fontsize=fs_intra, zorder=9)  # radial ticks

print('\n\n')

# %% HOLOGRAPHIC TWO PHOTON STIM EXCITABILITY ################################################

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis, ExpSeizureResults
from _main_.Post4apMain import Post4ap
from _utils_.io import import_expobj

from pycircstat.tests import vtest

results: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()
main = PhotostimAnalysisSlmTargets

sz_results: ExpSeizureResults = ExpSeizureResults.load()

################################################
# M) splitting responses during interictal phases ################################################
################################################
results = main.collect__interictal_responses_split(rerun=0)

ax = axes['L'][0]  #: interictal split - z scores

# figure for individual targets pooled across exps:
preictal_responses = flattenOnce(results.interictal_responses['preictal_responses'])
postictal_responses = flattenOnce(results.interictal_responses['postictal_responses'])
interictal_responses = flattenOnce(results.interictal_responses['very_interictal_responses'])

to_plot = [interictal_responses, preictal_responses, postictal_responses]

plot_bar_with_points(data=to_plot, bar=True, title='', fontsize=10, points_lw=0.5, points=False,
                     x_tick_labels=['All', 'Pre-ictal', 'Post-ictal'], colors=['gold', 'lightseagreen', 'lightcoral'],
                     y_label='Response magnitude\n(z-scored)', show=False, ylims=[-0.13, 0.43], lw=0.75,
                     alpha=1, fig=fig, ax=ax, s=15, capsize=4)

# 1-WAY ANOVA
stats.f_oneway(preictal_responses, interictal_responses, postictal_responses)

# create DataFrame to hold data
data_nums = []
num_pre = len(preictal_responses)
num_mid = len(interictal_responses)
num_post = len(postictal_responses)
data_nums.extend(['pre'] * num_pre)
data_nums.extend(['mid'] * num_mid)
data_nums.extend(['post'] * num_post)

df = pd.DataFrame({'score': preictal_responses + interictal_responses + postictal_responses,
                   'group': data_nums})

# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'],
                          alpha=0.05)

print(tukey)

# fig, ax = plt.subplots(figsize=(3,4))
data = [results.interictal_responses['preictal_responses'],
        results.interictal_responses['very_interictal_responses'],
        results.interictal_responses['postictal_responses']]

# run paired t test on preictal and postictal
t, p = stats.ttest_rel(interictal_responses, postictal_responses)
print(f"t = {t}, p = {p}")

# plot_bar_with_points(data=data, bar=False, title='', fontsize=10,points_lw=0.5,
#                      x_tick_labels=['Pre', 'interictal', 'Post'], colors=['lightseagreen', 'red', 'lightcoral'],
#                      y_label='Response magnitude\n(z-scored)', show=False, ylims=[-0.5, 0.8],
#                      alpha=1, fig=fig, ax=ax, s=15)


################################################
# H) ################################################
################################################

main = PhotostimAnalysisSlmTargets
main.plot_photostim_traces_stacked_LFP_pre4ap_post4ap(cells_to_plot='median10', y_spacing_factor=6,
                                                      fig=fig, ax_cat=(axes['Htop'], axes['Hbottom']))

################################################
# G) alloptical interrogation + experimental prep ################################################
################################################

ax = axes['G'][0]
#
sch_path = f'{fig_items}alloptical-interrogation-schematic.png'
img = mpimg.imread(sch_path)
ax.imshow(img, interpolation='none')
rfv.naked(ax)

################################################
# J) GRAND AVERAGE PHOTOSTIM TRACES AND AVERAGE ACROSS EXPERIMENTS ################################################
################################################

ax = axes['J'][0]

results.collect_grand_avg_alloptical_traces(rerun=0)

# all conditions - grand average photostim average of targets - make figure

# photostims - targets
avg_ = np.mean(results.grand_avg_traces['baseline'], axis=0)
sem_ = stats.sem(results.grand_avg_traces['baseline'], axis=0, ddof=1, nan_policy='omit')
ax.plot(results.grand_avg_traces['time_arr'], avg_, color='royalblue', lw=1)
ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
                color='royalblue')

avg_ = np.mean(results.grand_avg_traces['interictal'], axis=0)
sem_ = stats.sem(results.grand_avg_traces['interictal'], axis=0, ddof=1, nan_policy='omit')
ax.plot(results.grand_avg_traces['time_arr'], avg_, color='forestgreen', lw=1)
ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
                color='forestgreen')

# span over stim frames
stim_ = np.where(sem_ == 0)[0]
ax.axvspan(results.grand_avg_traces['time_arr'][stim_[0] - 1], results.grand_avg_traces['time_arr'][stim_[-1] + 2],
           color='lightcoral', zorder=5)
rfv.naked(ax)
ax.set_ylim([-1.5, 30])

rfv.add_scale_bar(ax=ax, length=(5, 1), bartype='L', text=('5%\ndFF', '1 s'), loc=(2.5, 20),
                  text_offset=[0.2, 3], fs=fs_intra)

ax.set_ylabel('All neuron targets', fontsize=fs_extra)
# ax.text(s='All neuron targets', x=-5, y=1.5, rotation=90, fontsize=fs_extra)
# ax.text(s='Photostimulation', x=0.058, y=0, rotation=90, fontsize=fs_intra, fontweight='bold', color='white', zorder=9)

################################################
# K) BAR PLOT OF AVG PHOTOSTIMULATION RESPONSE OF TARGETS ACROSS CONDITIONS ################################################
################################################

ax = axes['J'][1]

results.collect_avg_photostim_responses_states(rerun=0)
baseline_responses = results.avg_photostim_responses_all('baseline')
interictal_responses = results.avg_photostim_responses_all('interictal')
# ictal_responses = results.avg_photostim_responses['ictal']

# ttest of independence
ttest = stats.ttest_rel(baseline_responses, interictal_responses)
print(f"ttest paired - p(Baseline vs. interictal): {ttest[1]}")

# # kruskal wallis - for comparing baseline, interictal and ictal
# kw_score = stats.kruskal(baseline_responses,
#                interictal_responses,
#                ictal_responses)
# print(f"C': KW: n.s., p={kw_score[1]}")
# # 1-WAY ANOVA
# oneway_score = stats.f_oneway(baseline_responses,
#                interictal_responses,
#                ictal_responses)
# create DataFrame to hold data
# data_nums = []
# num_baseline = len(baseline_responses)
# num_interictal = len(interictal_responses)
# num_ictal = len(ictal_responses)
# data_nums.extend(['baseline'] * num_baseline)
# data_nums.extend(['interictal'] * num_interictal)
# data_nums.extend(['ictal'] * num_ictal)
#
# df = pd.DataFrame({'score': flattenOnce([baseline_responses, interictal_responses, ictal_responses]),
#                    'group': data_nums})
# # perform Tukey's test
# tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'],
#                           alpha=0.05)
# print(tukey)


fig, ax = plot_bar_with_points(data=[baseline_responses, interictal_responses],
                               bar=False, title='', x_tick_labels=['Baseline', 'Interictal'], points_lw=0.75,
                               colors=['royalblue', 'forestgreen'], figsize=(4, 4), y_label='% dFF', fontsize=fs_extra,
                               lw=1.3, capsize=4,
                               s=25, alpha=1, ylims=[-19, 90], show=False, fig=fig, ax=ax,
                               sig_compare_lines={'n.s.': [0, 1]})

################################################
# L) BAR PLOT OF AVG PHOTOSTIMULATION FOV RAW FLU ACROSS CONDITIONS ################################################
################################################


ax = axes['K'][0]

results.collect_avg_prestimf_states(rerun=0)
baseline_prestimf = results.avg_prestim_Flu_all('baseline')
interictal_prestimf = results.avg_prestim_Flu_all('interictal')

# ttest of independence
ttest = stats.ttest_rel(baseline_prestimf, interictal_prestimf)
print(f"ttest paired - p(Baseline vs. interictal): {ttest[1]}")

# ictal_prestimf = results.avg_prestim_Flu['ictal']


# kruskal wallis
# kw_score = stats.kruskal(baseline_prestimf, interictal_prestimf, ictal_prestimf)
# print(f"D: KW: *p<0.05, p={kw_score[1]}")

# 1-WAY ANOVA
# oneway_score = stats.f_oneway(baseline_prestimf, interictal_prestimf, ictal_prestimf)

# print(f"D: f_oneway: **p<0.01, p={oneway_score[1]}")

# create DataFrame to hold data
# data_nums = []
# num_baseline = len(baseline_prestimf)
# num_interictal = len(interictal_prestimf)
# num_ictal = len(ictal_prestimf)
# data_nums.extend(['baseline'] * num_baseline)
# data_nums.extend(['interictal'] * num_interictal)
# data_nums.extend(['ictal'] * num_ictal)

# df = pd.DataFrame({'score': flattenOnce([baseline_prestimf, interictal_prestimf, ictal_prestimf]),
#                    'group': data_nums})

# perform Tukey's test
# tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
# print(tukey)


plot_bar_with_points(data=[baseline_prestimf, interictal_prestimf],
                     bar=False, title='', show=False, fig=fig, ax=ax,
                     x_tick_labels=['Baseline', 'Interictal'], fontsize=ExpMetainfo.figures.fontsize['extraplot'],
                     colors=['royalblue', 'forestgreen'],
                     y_label='Pre-stim\nFluorescence (a.u.)', lw=1.3, points_lw=0.8,
                     ylims=[0, 1900], alpha=1, s=25, sig_compare_lines={'n.s.': [0, 1]})

print('\n\n')

################################################
# I) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period ################################################
################################################
# bbox = Bbox.from_extents(0.0, 0.05, 0.24, 0.19)
bbox = Bbox.from_extents(0.02, 0.08, 0.20, 0.16)
_axes = np.empty(shape=(1, 1), dtype=object)
ax = fig.add_subplot(projection='polar')
ax.set_position(pos=bbox)
# rfv.add_label_axes(text='E', ax=ax, y_adjust=0.02)
# fig.show()


# run data analysis
if not hasattr(sz_results, 'exp_sz_occurrence'):
    sz_results.exp_sz_occurrence, sz_results.total_sz_occurrence = ExpSeizureAnalysis.collectSzOccurrenceRelativeStim()
    sz_results.save_results()

expobj: Post4ap = import_expobj(exp_prep='RL109 t-013')

# make plot
bin_width = int(1 * expobj.fps) if expobj.fps == 15 else int(0.5 * expobj.fps)
# period = len(np.arange(0, (expobj.stim_interval_fr / bin_width)))
period = 10
theta = (2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period

# by experiment
# for exp, values in exp_sz_occurrence.items():
#     plot = values
#     ax.bar(theta, plot, width=(2 * np.pi) / period, bottom=0.0, alpha=0.5)

# across all seizures
total_sz = np.sum(np.sum(sz_results.total_sz_occurrence, axis=0))
sz_prob = np.sum(sz_results.total_sz_occurrence, axis=0) / total_sz

pval, z = vtest(alpha=theta, mu=0, w=np.sum(sz_results.total_sz_occurrence, axis=0))
print(f'pval for twoP stim seizure incidence is: {pval}')

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300, figsize=(3,3))

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color=ExpMetainfo.figures.colors['general'],
       edgecolor='black', lw=0.7)

ax.set_rmax(0.33)
ax.set_rticks([0.15, 0.3])  # Less radial ticks
ax.set_yticklabels(['0.15', '0.3'], fontsize=fs_intra)  # Less radial ticks
ax.set_rlabel_position(-55)  # Move radial labels away from plotted line
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels([r'$\it{\Theta}$ = 0', '', '', ''], fontsize=fs_extra)
ax.spines['polar'].set_visible(False)

ax.set_title(label='Seizure probability', fontsize=fs_extra, va='bottom')

# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.pdf")

fig.show()
