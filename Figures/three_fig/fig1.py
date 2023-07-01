# %% FIGURE 1 - LIVE IMAGING OF SEIZURES IN AWAKE ANIMALS (to be combined with Fig 1 from Inh serving as the bottom half)

import sys

import pandas as pd
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

from _exp_metainfo_.exp_metainfo import fontsize_intraplot


from _analysis_.sz_analysis._ClassSuite2pROIsSzAnalysis import Suite2pROIsSzResults
from _results_.sz4ap_results import plotHeatMapSzAllCells
from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureResults


from _utils_.io import import_expobj
from _utils_.alloptical_plotting import save_figure
import funcsforprajay.plotting as pplot
import numpy as np
from scipy import stats

from _main_.Post4apMain import Post4ap

import rep_fig_vis as rfv

fig_title = f'fig1_szimgA-E'
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/3fig/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'


results = Suite2pROIsSzResults.load()
results_seizure: ExpSeizureResults = ExpSeizureResults.load()

expobj: Post4ap = import_expobj(exp_prep='RL108 t-013')



# %% MAKE FIGURE LAYOUT

# set layout of the figure
layout = {
    'A': {'panel_shape': (1, 1),
          'bound': (0.05, 0.80, 0.33, 0.95)},
    'B': {'panel_shape': (1, 1),
          'bound': (0.43, 0.86, 0.95, 0.95)},
    'C top': {'panel_shape': (1, 1),
          'bound': (0.07, 0.67, 0.30, 0.72)},
    'C bottom': {'panel_shape': (1, 1),
          'bound': (0.07, 0.57, 0.30, 0.67)},
    'D': {'panel_shape': (3, 1),
          'bound': (0.40, 0.57, 0.65, 0.67),
          'wspace': 2},
    'E': {'panel_shape': (1, 1),
          'bound': (0.73, 0.57, 0.80, 0.67),
          'wspace': 1},
    ## F to I below are for the INH part of the figure
    'F': {'panel_shape': (1, 1),
          'bound': (0.05, 0.25, 0.15, 0.50)},
    'Gtop': {'panel_shape': (1, 1),
             'bound': (0.35, 0.25, 0.53, 0.48)},
    'Gbottom': {'panel_shape': (1, 1),
                'bound': (0.35, 0.20, 0.53, 0.25)},
    'H': {'panel_shape': (1, 1),
          'bound': (0.67, 0.38, 0.80, 0.48)},
    'I': {'panel_shape': (1, 1),  # for LFP signal for seizure - removed for now
          'bound': (0.67, 0.20, 0.80, 0.30)},
}


test = 0
save_fig = True if not test > 0 else False
dpi = 80 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.


rfv.naked(axes['A'][0])
# rfv.naked(axes['B'][0])
rfv.naked(axes['B'][0])
rfv.naked(axes['C bottom'][0])
rfv.naked(axes['C top'][0])
# rfv.add_label_axes(text='A', ax=axes['A'][0], y_adjust=0, x_adjust=0.04)
# rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0)
# rfv.add_label_axes(text='B', ax=axes['B'][0], y_adjust=0, x_adjust=0.07)
# rfv.add_label_axes(text='C', ax=axes['C top'][0], x_adjust=0.06, y_adjust=-0.02)

print('\n\n')




# %% D) seizure stats

from _analysis_.sz_analysis._ClassExpSeizureAnalysis import ExpSeizureAnalysis as main, ExpSeizureResults

# main.FOVszInvasionTime()
# main.calc__szInvasionTime()
# main.plot__sz_invasion()

ax, ax2, ax3 = axes['D'][0], axes['D'][1], axes['D'][2]

main.plot__sz_propagation_speed(results=results_seizure, fig=fig, ax=ax3, show=False)
main.plot__sz_incidence(fig=fig, ax=ax, show=False)
main.plot__sz_lengths(fig=fig, ax=ax2, show=False)



ax.set_ylabel(f'Ictal events / min', fontsize=fontsize_intraplot, labelpad=0)
ax.set_yticks([0, 1], [0, 1], fontsize=fontsize_intraplot)
ax.set_title('')
ax2.set_title('')
ax2.set_ylabel('Length (secs)', fontsize=fontsize_intraplot, labelpad=0)
ax2.set_yticks([0, 120], [0, 120], fontsize=fontsize_intraplot)
ax2.set_ylim([0, 120])
ax3.set_ylabel('Speed ($\mu$$\it{m}$/sec)', fontsize=fontsize_intraplot, labelpad=0)
ax3.set_yticks([0, 40], [0, 40], fontsize=fontsize_intraplot)
ax3.set_ylim([0, 40])
# rfv.add_label_axes(text='D', ax=ax, x_adjust=0.07, y_adjust=0.03)


# %% E) Activity rates across seizure imaging
ax4 = axes['E'][0]
results = Suite2pROIsSzResults.load()


baseline_rates = [np.mean(i) for i in results.neural_activity_rate['baseline']]
interictal_rates = [np.mean(i) for i in results.neural_activity_rate['interictal']]
ictal_rates = [np.mean(i) for i in results.neural_activity_rate['ictal']]

activity_rates = [baseline_rates, interictal_rates, ictal_rates]

pplot.plot_bar_with_points(
    data=[baseline_rates, interictal_rates, ictal_rates],
    bar=True, x_tick_labels=['Baseline', 'Interictal', 'Ictal'], points=False,
    colors=['cornflowerblue', 'forestgreen', 'purple'], lw=0.75, title='', y_label='Actvity rate', alpha=0.7,
    ylims=[0, 200], show=False, capsize=1.5, fontsize=fontsize_intraplot, width_factor=1.15,
    ax=ax4)

# rfv.add_label_axes(text='E', ax=ax4, x_adjust=0.07, y_adjust=0.03)

# fig.show(); print('a')
oneway_r = stats.f_oneway(*activity_rates)

print(f"1-way ANOVA  of baseline, interictal and ictal activity rates: {oneway_r.pvalue}")

# create DataFrame to hold data
data_nums = []
num_baseline = len(baseline_rates)
num_interictal = len(interictal_rates)
num_ictal = len(ictal_rates)
data_nums.extend(['baseline'] * num_baseline)
data_nums.extend(['interictal'] * num_interictal)
data_nums.extend(['ictal'] * num_ictal)

df = pd.DataFrame({'score': baseline_rates + interictal_rates + ictal_rates,
                   'group': data_nums})

# perform Tukey's test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'],
                          alpha=0.05)

print(tukey)



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
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.pdf")


fig.show()


