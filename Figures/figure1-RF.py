## FIGURE 1 - LIVE IMAGING OF SEIZURES IN AWAKE ANIMALS
import sys

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

from _analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis import Suite2pROIsSz, Suite2pROIsSzResults
from _results_.sz4ap_results import plotHeatMapSzAllCells

from _utils_.io import import_expobj
from alloptical_utils_pj import save_figure
import numpy as np

from _main_.Post4apMain import Post4ap

import rep_fig_vis as rfv

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

results = Suite2pROIsSzResults.load()

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')


# %% MAKE FIGURE LAYOUT
rfv.set_fontsize(6)
dpi = 100
save_fig = True


# set layout of the figure
layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.05, 0.80, 0.33, 0.95)},
    'B': {'panel_shape': (1, 1),
          'bound': (0.05, 0.70, 0.33, 0.76)},
    'C': {'panel_shape': (1, 1),
          'bound': (0.35, 0.86, 0.95, 0.95)},
    'D top': {'panel_shape': (1, 1),
          'bound': (0.37, 0.78, 0.60, 0.83)},
    'D bottom': {'panel_shape': (1, 1),
          'bound': (0.37, 0.68, 0.60, 0.78)},
    'E': {'panel_shape': (2, 1),
          'bound': (0.70, 0.70, 0.82, 0.78),
          'wspace': 2.5},
    'F': {'panel_shape': (1, 1),
          'bound': (0.90, 0.70, 0.95, 0.78)
          }
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)

rfv.naked(axes['A'][0])
rfv.naked(axes['B'][0])
rfv.naked(axes['C'][0])
rfv.naked(axes['D bottom'][0])
rfv.naked(axes['D top'][0])
rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0)
rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0)
rfv.add_label_axes(text='C', ax=axes['C'][0], y_adjust=0)

print('\n\n')



# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.



# %% D) suite2p cells gcamp imaging for seizures, with simultaneous LFP recording

# fig, axs = plt.subplots(2, 1, figsize=(6, 6))
# fig, axs[0] = aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='time', fig=fig, ax=axs[0], show=False)
# fig, axs[1] = aoplot.plotLfpSignal(expobj=expobj, stim_span_color='', x_axis='time', fig=fig, ax=axs[1], show=False)
# axs[0].set_xlim([400 * expobj.fps, 470 * expobj.fps])
# axs[1].set_xlim([400 * expobj.paq_rate, 470 * expobj.paq_rate])
# fig.show()

# plot heatmap of raw neuropil corrected s2p signal from s2p cells
time = (400, 460)
frames = (time[0] * expobj.fps, time[1] * expobj.fps)
paq = (time[0] * expobj.paq_rate, time[1] * expobj.paq_rate)

ax1, ax2 = axes['D bottom'][0], axes['D top'][0]
plotHeatMapSzAllCells(expobj=expobj, sz_num=4, ax1=ax1, ax2=ax2, fig=fig)
ax1.set_yticks([0, 100])

x = ax2.get_xlim()[1]

rfv.add_label_axes(text='D', ax=axes['D top'][0], x_adjust=0.05, y_adjust=-0.02)



# %% E) seizure stats

from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis as main

# main.FOVszInvasionTime()
# main.calc__szInvasionTime()
# main.plot__sz_invasion()

ax, ax2 = axes['E'][0], axes['E'][1]

main.plot__sz_incidence(fig=fig, ax=ax)
main.plot__sz_lengths(fig=fig, ax=ax2)

ax.set_ylabel('')
ax.set_yticks([0, 1])
ax.set_yticklabels([0, 1])
ax2.set_ylabel('')
ax.set_title('')
ax2.set_yticks([0, 100])
ax2.set_title('')
rfv.add_label_axes(text='E', ax=axes['E'][0], x_adjust=0.07, y_adjust=0.03)



# %% F) DECONVOLVED SPIKE RATE ANALYSIS


# RUN collect spk rates
Suite2pROIsSz.collect__avg_spk_rate(Suite2pROIsSzResults=results, rerun=False)


# PLOT averaged results as a bar chart

# results = Suite2pROIsSzResults.load()
#
# # todo add statistical test for this bar chart!
# Suite2pROIsSz.plot__avg_spk_rate(results.avg_spk_rate['baseline'],
#                                  results.avg_spk_rate['interictal'])
#

# PLOT individual results as a cum sum plot

# evaluate the histogram
ax = axes['F'][0]
# f, ax2 = plt.subplots(figsize=(5, 5))
# baseline experiments
for pre4ap_exp in results.neural_activity_rate['baseline']:
    # test plot cumsum plot
    values, base = np.histogram(pre4ap_exp, bins=100)

    # ax2.hist(pre4ap_exp, density=True, histtype='stepfilled', alpha=0.3, bins=100, color='cornflowerblue')

    # evaluate the cumulative function
    cumulative = np.cumsum(values) / len(pre4ap_exp)

    # plot the cumulative function
    ax.plot(base[:-1], cumulative, c='navy', alpha=0.5, lw=1)

# baseline experiments
for interictal_exp in results.neural_activity_rate['interictal']:
    # test plot cumsum plot
    values, base = np.histogram(interictal_exp, bins=100)

    # ax2.hist(interictal_exp, density=True, histtype='stepfilled', alpha=0.3, bins=100, color='forestgreen')

    # evaluate the cumulative function
    cumulative = np.cumsum(values) / len(interictal_exp)

    # plot the cumulative function
    ax.plot(base[:-1], cumulative, c='darkgreen', alpha=0.5, lw=1)

ax.set_xlim([0, 200])
# ax2.set_xlim([0, 200])
ax.set_yticks([0, 1])
ax.set_xticks([0, 100, 200])
# ax.set_xlabel('Avg. activity\nrate (Hz)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

rfv.add_label_axes(text='F', ax=axes['F'][0], y_adjust=0.03, x_adjust=0.06)


# %%

if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure1-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure1-RF.svg")


fig.show()


