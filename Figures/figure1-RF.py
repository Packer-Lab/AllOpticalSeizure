## FIGURE 1 - LIVE IMAGING OF SEIZURES IN AWAKE ANIMALS
import sys

from matplotlib import pyplot as plt

from _exp_metainfo_.exp_metainfo import fontsize_extraplot

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

from _analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis import Suite2pROIsSz, Suite2pROIsSzResults
from _results_.sz4ap_results import plotHeatMapSzAllCells
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureResults


from _utils_.io import import_expobj
from _utils_.alloptical_plotting import save_figure
import numpy as np

from _main_.Post4apMain import Post4ap

import rep_fig_vis as rfv

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

results = Suite2pROIsSzResults.load()
results_seizure: ExpSeizureResults = ExpSeizureResults.load()

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')


# %% MAKE FIGURE LAYOUT
rfv.set_fontsize(fontsize_extraplot)

test = 0
save_fig = True if not test else False
dpi = 120 if test else 300

# set layout of the figure
layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.05, 0.80, 0.33, 0.95)},
    # 'B': {'panel_shape': (1, 1),
    #       'bound': (0.05, 0.70, 0.33, 0.76)},
    'B': {'panel_shape': (1, 1),
          'bound': (0.43, 0.86, 0.95, 0.95)},
    'C top': {'panel_shape': (1, 1),
          'bound': (0.07, 0.67, 0.30, 0.72)},
    'C bottom': {'panel_shape': (1, 1),
          'bound': (0.07, 0.57, 0.30, 0.67)},
    'D': {'panel_shape': (3, 1),
          'bound': (0.43, 0.57, 0.73, 0.67),
          'wspace': 1.8},
    # 'E': {'panel_shape': (1, 1),
    #       'bound': (0.90, 0.70, 0.95, 0.78)
    #       }
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)

rfv.naked(axes['A'][0])
# rfv.naked(axes['B'][0])
rfv.naked(axes['B'][0])
rfv.naked(axes['C bottom'][0])
rfv.naked(axes['C top'][0])
rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0, x_adjust=0.04)
# rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0)
rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0, x_adjust=0.07)
rfv.add_label_axes(text='C', ax=axes['C top'][0], x_adjust=0.06, y_adjust=-0.02)

print('\n\n')


# rfv.show_test_figure_layout(fig, axes=axes, show=True)  # test what layout looks like quickly, but can also skip and moveon to plotting data.



# %% D) seizure stats

from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis as main, ExpSeizureResults

# main.FOVszInvasionTime()
# main.calc__szInvasionTime()
# main.plot__sz_invasion()

ax, ax2, ax3 = axes['D'][0], axes['D'][1], axes['D'][2]

main.plot__sz_incidence(fig=fig, ax=ax, show=False)
main.plot__sz_lengths(fig=fig, ax=ax2, show=False)
main.plot__sz_propagation_speed(results=results_seizure, fig=fig, ax=ax3, show=False)

ax.set_ylabel(f'Ictal events / min', fontsize=fontsize_extraplot)
ax.set_yticks([0, 1], [0, 1], fontsize=fontsize_extraplot)
ax.set_title('')
ax2.set_title('')
ax2.set_ylabel('Length (secs)', fontsize=fontsize_extraplot)
ax2.set_yticks([0, 120], [0, 120], fontsize=fontsize_extraplot)
ax2.set_ylim([0, 120])
ax3.set_ylabel('Speed ($\mu$$\it{m}$/sec)', fontsize=fontsize_extraplot)
ax3.set_yticks([0, 40], [0, 40], fontsize=fontsize_extraplot)
ax3.set_ylim([0, 40])
rfv.add_label_axes(text='D', ax=axes['D'][0], x_adjust=0.07, y_adjust=0.03)



# %% C) suite2p cells gcamp imaging for seizures, with simultaneous LFP recording

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

ax1, ax2 = axes['C bottom'][0], axes['C top'][0]
plotHeatMapSzAllCells(expobj=expobj, sz_num=4, ax1=ax1, ax2=ax2, fig=fig)
ax1.set_yticks([0, 100])

x = ax2.get_xlim()[1]





# %%

if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure1-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure1-RF.pdf")


fig.show()


