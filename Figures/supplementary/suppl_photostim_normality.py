"""
TODO:

make plot code

use: scipy.stats.normaltest() on single- target/trial photostim responses across experiments


A: The overall distribution of dFF photostimulation responses across all targeted neurons and photostimulation trials follows a normal distribution.
B: The photostimulation responses of targeted neurons pass normality tests in all experiments.

"""

# %%
import sys

import numpy as np
from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_hist_density
from scipy import stats

from _exp_metainfo_.exp_metainfo import AllOpticalExpsToAnalyze
from _main_.AllOpticalMain import alloptical
from _utils_.io import import_expobj

import matplotlib.pyplot as plt
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv

from alloptical_utils_pj import save_figure

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

fontsize = 8
fs = fontsize
rfv.set_fontsize(fs)

save_fig = True

# %% MAKE FIGURE LAYOUT
layout = {
    'left': {'panel_shape': (1, 1),
             'bound': (0.15, 0.725, 0.40, 0.85)},
    'right': {'panel_shape': (1, 1),
                    'bound': (0.58, 0.70, 0.88, 0.90)},
}

dpi = 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)

# rfv.show_test_figure_layout(fig, axes=axes, show=1)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% SCHEMATIZED NORMAL DISTRIBUTION

ax = axes['left'][0]

x = np.arange(0, 100)
y = np.random.standard_normal(10000)*0.4
plot_hist_density(data=[list(y)], mean_line=False, num_bins=101, ax=ax, fig=fig, show=False, fill_alpha=0.2,
                              lw=1, fill_color=['gray'], bar_alpha=0.0)
ax.set_title('')
ax.set_ylabel('')
# ax.set_xlabel(f'Normalized\nphotostimulation responses')
ax.set_xticks([0])
ax.set_yticks([])
ax.set_ylim([-0.08, 1.1])
ax.set_xlim([-1, 1])
ax.set_xticklabels([0])
rfv.despine(ax, remove=['top', 'left', 'right'])
# ax.plot(x, y, lw=3)
# ax.fill_between(x, 0, y, color='skyblue')

# %% ANALYSIS + PLOTTING OF NORMALITY OF PHOTOSTIMULATION RESPONSES FROM EXPS

p_value_normal = {}

ax=axes['right'][0]
for trial in flattenOnce(AllOpticalExpsToAnalyze.pre_4ap_trials):
        if trial != 'PS04 t-017': # exclude doubled up trial from same mouse
            expobj: alloptical = import_expobj(exp_prep=trial)
            responses = flattenOnce(expobj.PhotostimResponsesSLMTargets.adata.X) - np.mean(expobj.PhotostimResponsesSLMTargets.adata.X)
            plot_hist_density(data=[list(responses)], mean_line=False, num_bins=21, ax=ax, fig=fig, show=False, fill_alpha=0.1, bar_alpha=0.5,
                              lw=1)
            p_value_normal[expobj.t_series_name] = stats.normaltest(responses)[1]

ax.set_title(f'All experiments\n(Baseline trials)', fontsize=12)
ax.set_ylabel('Density', fontsize=10)
ax.set_xlabel('Photostimulation responses\n(mean subtracted, % dFF)', fontsize=10)
ax.set_xlim([-200, 200])


print(p_value_normal)
print(p_value_normal.values())

if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure-suppl-normality-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure-suppl-normality-RF.svg")

fig.show()


