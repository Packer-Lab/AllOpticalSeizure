"""
two photon stim - interictal
"""


# %%
import sys; sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure'])

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(['alloptical', 'use_mathtext'])

from funcsforprajay.plotting.plotting import plot_bar_with_points
from funcsforprajay.funcs import flattenOnce
from matplotlib.transforms import Bbox
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %%
from _exp_metainfo_.exp_metainfo import ExpMetainfo

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis, ExpSeizureResults
from _main_.Post4apMain import Post4ap
from _utils_.io import import_expobj
from _utils_.alloptical_plotting import save_figure

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.image as mpimg
import rep_fig_vis as rfv

from pycircstat.tests import vtest

results: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()
main = PhotostimAnalysisSlmTargets



sz_results: ExpSeizureResults = ExpSeizureResults.load()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft_Apr2023/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'


# %%

## Set general plotting parameters
fs = ExpMetainfo.figures.fontsize['extraplot']
rfv.set_fontsize(fs)

colour_list = ['#101820', '#1b362c', '#2f553d', '#4f7553', '#79936f', '#aeae92']
colours_misc_dict = {xx: colour_list[xx] for xx in range(len(colour_list))}

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'main-left': {'panel_shape': (1, 1),
                  'bound': (0.05, 0.64, 0.30, 0.98)},
    'toprighttop': {'panel_shape': (2, 1),
                    'bound': (0.31, 0.86, 0.90, 0.95)},
    'toprightbottom': {'panel_shape': (2, 1),
                       'bound': (0.31, 0.64, 0.90, 0.85)},
    'bottom-C': {'panel_shape': (2, 1),
                 'bound': (0.05, 0.47, 0.38, 0.60),
                 'wspace': 0.8},
    'bottom-D': {'panel_shape': (1, 1),
                 'bound': (0.47, 0.47, 0.59, 0.60),
                 'wspace': 0.4},
    'bottom-right-F': {'panel_shape': (1,1),
                        'bound': (0.85, 0.47, 0.95, 0.60)
                       }
    }


test = 1
save_fig = True if not test > 0 else False
dpi = 150 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% add labels
rfv.add_label_axes(text="D", ax=axes['bottom-D'][0], y_adjust=0, x_adjust=0.11)
rfv.add_label_axes(text="C'", ax=axes['bottom-C'][1], y_adjust=0, x_adjust=0.07)
rfv.add_label_axes(text="C", ax=axes['bottom-C'][0], y_adjust=0.0)



# %% F - splitting responses during interictal phases
results = main.collect__interictal_responses_split(rerun=0)


# %%

ax = axes['bottom-right-F'][0]  #: interictal split - z scores


# figure for individual targets pooled across exps:
preictal_responses = flattenOnce(results.interictal_responses['preictal_responses'])
postictal_responses = flattenOnce(results.interictal_responses['postictal_responses'])
interictal_responses = flattenOnce(results.interictal_responses['very_interictal_responses'])

to_plot = [interictal_responses, preictal_responses, postictal_responses]

plot_bar_with_points(data=to_plot, bar=True, title='', fontsize=10,points_lw=0.5,
                     x_tick_labels=['All', 'Pre-ictal', 'Post-ictal'], colors=['gold', 'lightseagreen', 'lightcoral'],
                     y_label='Response magnitude\n(z-scored)', show=False, ylims=[-0.5, 0.8],
                     alpha=1, fig=fig, ax=ax, s=15)



# %%
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

rfv.add_label_axes(text='F', ax=ax, x_adjust=0.12)



# %% B + B')
ax = axes['toprighttop'][0]
rfv.add_label_axes(text='B', ax=ax, y_adjust=0.01)
# fig.show()

# ax = axes['toprightbottom'][0]
# rfv.add_label_axes(text="B'", ax=ax, y_adjust=-0.01)

main = PhotostimAnalysisSlmTargets
main.plot_photostim_traces_stacked_LFP_pre4ap_post4ap(fig=fig, ax_cat=(axes['toprighttop'], axes['toprightbottom']))



# %% A) alloptical interrogation + experimental prep
ax = axes['main-left'][0]
rfv.add_label_axes(text="A", ax=ax, y_adjust=-0.02)
#
sch_path = f'{fig_items}alloptical-interrogation-schematic.png'
img = mpimg.imread(sch_path)
ax.imshow(img, interpolation='none')
# ax.text(s='All-optical \ninterrogation', x = 12, y=85, fontsize=10, fontweight='semibold', ha='center', va='center')
rfv.naked(ax)


# %% C) GRAND AVERAGE PHOTOSTIM TRACES AND AVERAGE ACROSS EXPERIMENTS

ax = axes['bottom-C'][0]

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

# avg_ = np.mean(results.grand_avg_traces['ictal'], axis=0)
# sem_ = stats.sem(results.grand_avg_traces['ictal'], axis=0, ddof=1, nan_policy='omit')
# ax.plot(results.grand_avg_traces['time_arr'], avg_, color='blueviolet', lw=1)
# ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
#                 color='purple')

# # fakestims - targets
# avg_ = np.mean(fakestim_targets_average_traces, axis=0)
# std_ = np.std(fakestim_targets_average_traces, axis=0, ddof=1)
# ax.plot(time_arr, avg_, color='black', lw=1.5)
# ax.fill_between(x=time_arr, y1=avg_ + std_, y2=avg_ - std_, alpha=0.3, zorder=2, color='gray')

# span over stim frames
stim_ = np.where(sem_ == 0)[0]
ax.axvspan(results.grand_avg_traces['time_arr'][stim_[0] - 1], results.grand_avg_traces['time_arr'][stim_[-1] + 2],
           color='lightcoral', zorder=5)
rfv.naked(ax)
ax.set_ylim([-1.5, 30])

rfv.add_scale_bar(ax=ax, length=(5, 1), bartype='L', text=('5%\ndFF', '1$\it{sec}$'), loc=(2.5, 20),
                  text_offset=[0.2, 3], fs=ExpMetainfo.figures.fontsize['intraplot'])

ax.text(s='All neuron targets', x = -2, y=1.5, rotation=90, fontsize=fs)
ax.text(s='Photostimulation', x = 0.06, y=0, rotation=90, fontsize=ExpMetainfo.figures.fontsize['intraplot'], fontweight='bold', color='white', zorder=9)

# ax.set_ylabel('dFF')
# ax.set_xlabel('Time (secs) rel. to stim')
# ax.set_title('grand average all cells, all exps - baseline', wrap=True, fontsize='small')


# %% C') BAR PLOT OF AVG PHOTOSTIMULATION RESPONSE OF TARGETS ACROSS CONDITIONS

ax = axes['bottom-C'][1]

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
                     bar=False, title='', x_tick_labels=['Baseline', 'Interictal'], points_lw=0.8,
                     colors=['royalblue', 'forestgreen'], figsize=(4, 4), y_label='% dFF', fontsize=fs, lw=1.3,
                     s=25, alpha=1, ylims=[-19, 90], show=False, fig=fig, ax=ax, sig_compare_lines={'n.s.': [0, 1]})



# %% D) BAR PLOT OF AVG PHOTOSTIMULATION FOV RAW FLU ACROSS CONDITIONS

ax = axes['bottom-D'][0]

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



# %% E) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period
bbox = Bbox.from_extents(0.69, 0.445, 0.83, 0.595)
_axes = np.empty(shape=(1, 1), dtype=object)
ax = fig.add_subplot(projection='polar')
ax.set_position(pos=bbox)
rfv.add_label_axes(text='E', ax=ax, y_adjust=0.02)
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

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color=ExpMetainfo.figures.colors['general'], edgecolor='black', lw=0.7)

ax.set_rmax(1.1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # Less radial ticks
ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=8)  # Less radial ticks
ax.set_rlabel_position(-60)  # Move radial labels away from plotted line
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels([r'$\it{\Theta}$ = 0', '', '', ''], fontsize=10)
ax.spines['polar'].set_visible(False)

ax.set_title(label='Seizure probability', fontsize=fs, va='bottom')

# fig.show()


# %% add plots to axes
if save_fig and dpi >= 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure3-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure3-RF.svg")

fig.show()


