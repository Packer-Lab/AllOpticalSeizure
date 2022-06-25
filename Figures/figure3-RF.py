"""
TODO:

"""

# %%
import sys

from funcsforprajay.plotting.plotting import plot_bar_with_points
from matplotlib.transforms import Bbox
from scipy import stats

from _alloptical_utils import run_for_loop_across_exps
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_.alloptical_plotting import plot_settings
from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure

import cairosvg
from PIL import Image
from io import BytesIO

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rep_fig_vis as rfv

plot_settings()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

# %%

## Set general plotting parameters
rfv.set_fontsize(7)

colour_list = ['#101820', '#1b362c', '#2f553d', '#4f7553', '#79936f', '#aeae92']
colours_misc_dict = {xx: colour_list[xx] for xx in range(len(colour_list))}

save_fig = True

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
                 'bound': (0.05, 0.40, 0.40, 0.57),
                 'wspace': 0.4},
    'bottom-D': {'panel_shape': (1, 1),
                 'bound': (0.52, 0.40, 0.66, 0.57),
                 'wspace': 0.4}
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)

rfv.naked(axes['main-left'][0])

# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% add plots to axes ########################################################################################################################################################################################################################
# %% D) BAR PLOT OF AVG PHOTOSTIMULATION FOV RAW FLU ACROSS CONDITIONS
results: PhotostimResponsesSLMtargetsResults = PhotostimResponsesSLMtargetsResults.load()

ax = axes['bottom-D'][0]
rfv.add_label_axes(s="D", ax=ax, y_adjust=0, x_adjust=0.1)

results.collect_avg_prestimf_states(rerun=0)
baseline_prestimf = results.avg_prestim_Flu['baseline']
interictal_prestimf = results.avg_prestim_Flu['interictal']
ictal_prestimf = results.avg_prestim_Flu['ictal']

plot_bar_with_points(data=[baseline_prestimf, interictal_prestimf, ictal_prestimf],
                     bar=False, title='', show=False, fig=fig, ax=ax,
                     x_tick_labels=['Baseline', 'Interictal', 'Ictal'],
                     colors=['royalblue', 'mediumseagreen', 'blueviolet'],
                     y_label='Fluorescence (a.u.)',
                     ylims=[0, 1900], alpha=1, s=35)

# %% B + B')
ax = axes['toprighttop'][0]
rfv.add_label_axes(s='B', ax=ax, y_adjust=-0.01)
# fig.show()

# ax = axes['toprightbottom'][0]
# rfv.add_label_axes(s="B'", ax=ax, y_adjust=-0.01)

main = PhotostimAnalysisSlmTargets
main.plot_photostim_traces_stacked_LFP_pre4ap_post4ap(ax_cat=(axes['toprighttop'], axes['toprightbottom']), fig=fig)

# fig.show()

# %% F) Radial plot of Mean FOV for photostimulation trials, with period equal to that of photostimulation timing period
bbox = Bbox.from_extents(0.72, 0.41, 0.87, 0.56)
_axes = np.empty(shape=(1, 1), dtype=object)
ax = fig.add_subplot(projection='polar')
ax.set_position(pos=bbox)
rfv.add_label_axes(s='F', ax=ax, y_adjust=0.025)
# fig.show()


# run data analysis
exp_sz_occurrence, total_sz_occurrence = ExpSeizureAnalysis.collectSzOccurrenceRelativeStim()

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
total_sz = np.sum(np.sum(total_sz_occurrence, axis=0))
sz_prob = np.sum(total_sz_occurrence, axis=0) / total_sz

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300, figsize=(3,3))

ax.bar(theta, sz_prob, width=(2 * np.pi) / period, bottom=0.0, alpha=1, color='crimson')

ax.set_rmax(1.1)
ax.set_rticks([0.25, 0.5, 0.75, 1])  # Less radial ticks
ax.set_rlabel_position(-60)  # Move radial labels away from plotted line
ax.grid(True)
# ax.set_xticks((2 * np.pi) * np.arange(0, (expobj.stim_interval_fr / bin_width)) / period)
ax.set_xticks([0, (2 * np.pi) / 4, (2 * np.pi) / 2, (6 * np.pi) / 4])
ax.set_xticklabels(['0', '', '50', ''])
# ax.set_title("sz probability occurrence (binned every 1s)", va='bottom')
ax.spines['polar'].set_visible(False)

# fig.show()


# %% A) alloptical interrogation + experimental prep
ax = axes['main-left'][0]
rfv.add_label_axes(s="A", ax=ax, y_adjust=-0.015)
#
sch_path = f'{fig_items}alloptical-interrogation-schematic.png'
img = mpimg.imread(sch_path)
ax.imshow(img, interpolation='none')
# fig.show()


# %% C - C') GRAND AVERAGE PHOTOSTIM TRACES AND AVERAGE ACROSS EXPERIMENTS

ax = axes['bottom-C'][0]
rfv.add_label_axes(s="C", ax=ax, y_adjust=0.0)

results.collect_grand_avg_alloptical_traces(rerun=0)

# %% C - C') all conditions - grand average photostim average of targets - make figure

# photostims - targets
avg_ = np.mean(results.grand_avg_traces['baseline'], axis=0)
sem_ = stats.sem(results.grand_avg_traces['baseline'], axis=0, ddof=1, nan_policy='omit')
ax.plot(results.grand_avg_traces['time_arr'], avg_, color='royalblue', lw=1)
ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
                color='royalblue')

avg_ = np.mean(results.grand_avg_traces['interictal'], axis=0)
sem_ = stats.sem(results.grand_avg_traces['interictal'], axis=0, ddof=1, nan_policy='omit')
ax.plot(results.grand_avg_traces['time_arr'], avg_, color='mediumseagreen', lw=1)
ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
                color='forestgreen')

avg_ = np.mean(results.grand_avg_traces['ictal'], axis=0)
sem_ = stats.sem(results.grand_avg_traces['ictal'], axis=0, ddof=1, nan_policy='omit')
ax.plot(results.grand_avg_traces['time_arr'], avg_, color='blueviolet', lw=1)
ax.fill_between(x=results.grand_avg_traces['time_arr'], y1=avg_ + sem_, y2=avg_ - sem_, alpha=0.3, zorder=2,
                color='purple')

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
# ax.set_ylabel('dFF')
# ax.set_xlabel('Time (secs) rel. to stim')
# ax.set_title('grand average all cells, all exps - baseline', wrap=True, fontsize='small')

# %% C') BAR PLOT OF AVG PHOTOSTIMULATION RESPONSE OF TARGETS ACROSS CONDITIONS

ax = axes['bottom-C'][1]
rfv.add_label_axes(s="C'", ax=ax, y_adjust=0, x_adjust=0.07)

results.collect_avg_photostim_responses_states(rerun=0)
baseline_responses = results.avg_photostim_responses['baseline']
interictal_responses = results.avg_photostim_responses['interictal']
ictal_responses = results.avg_photostim_responses['ictal']

plot_bar_with_points(data=[baseline_responses, interictal_responses, ictal_responses],
                     bar=False, title='',
                     x_tick_labels=['Baseline', 'Interictal', 'Ictal'],
                     colors=['royalblue', 'mediumseagreen', 'blueviolet'], figsize=(4, 4), y_label='dFF',
                     s=35, alpha=1, ylims=[-19, 90], show=False, fig=fig, ax=ax)

# %%
# fig.show()

# %% add plots to axes
if save_fig:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure3-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure3-RF.svg")
