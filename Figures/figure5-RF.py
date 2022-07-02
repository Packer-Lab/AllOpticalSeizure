"""
TODO:

[ ] add schematic of seizure distance to target

suppl figure: write up RF code
- seizure boundary classification
- responses vs. pre-stim Flu targets annulus, across all conditions
    + stats tests on the averages? or can tests be carried out directly on the 2-D density plots?

"""

# %%
import sys

from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal, TargetsSzInvasionTemporalResults
from _analysis_.run__TargetsSzInvasionTemporal import plot__targets_sz_invasion_meantraces

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import rep_fig_vis as rfv
from _utils_.rfv_funcs import make_fig_layout, show_test_figure_layout, add_label_axes
import alloptical_utils_pj as Utils
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets

import funcsforprajay.plotting as pplot
import funcsforprajay.funcs as pj

import numpy as np
import matplotlib.image as mpimg

from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview, \
    TargetsSzInvasionSpatialResults_codereview

SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-sz-distance/"

main_spatial = TargetsSzInvasionSpatial_codereview
results_spatial = TargetsSzInvasionSpatialResults_codereview.load()

main_temporal = TargetsSzInvasionTemporal
results_temporal = TargetsSzInvasionTemporalResults.load()

from _utils_.alloptical_plotting import plot_settings

plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

# %% SETUP
## Set general plotting parameters
# rfv.set_fontsize(7)

## Set parameters
n_cols = 2
n_rows = 2

save_fig = True

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'main-left-top': {'panel_shape': (1, 1),
                      'bound': (0.05, 0.80, 0.27, 0.95)},
    'main-left-bottom': {'panel_shape': (1, 1),
                         'bound': (0.08, 0.45, 0.26, 0.60)},
    'main-right-tophigh': {'panel_shape': (1, 1),
                           'bound': (0.40, 0.75, 0.90, 0.95),
                           'wspace': 0.8},
    'main-right-toplow': {'panel_shape': (1, 1),
                          'bound': (0.40, 0.65, 0.90, 0.70),
                          'wspace': 0.8},
    'main-right-bottomhigh': {'panel_shape': (1, 1),
                              'bound': (0.40, 0.40, 0.90, 0.60),
                              'wspace': 0.8},
    'main-right-bottomlow': {'panel_shape': (1, 1),
                             'bound': (0.40, 0.30, 0.90, 0.35),
                             'wspace': 0.8},
}
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)

# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

x_adj = 0.09

# %% A' - photostim responses relative to distance to seizure
ax = axes['main-right-tophigh'][0]
rfv.add_label_axes(text="A'", ax=ax, x_adjust=x_adj + 0.03)
main_spatial.collect__binned__distance_v_responses(results=results_spatial, rerun=0)
results_spatial = TargetsSzInvasionSpatialResults_codereview.load()
main_spatial.plot__responses_v_distance_no_normalization(results=results_spatial, axes=(axes['main-right-tophigh'], axes['main-right-toplow']), fig=fig)
ax.text(x=50, y=2, s=f'{results_spatial.binned__distance_vs_photostimresponses["kruskal - binned responses"]}', fontsize=5)


# %% MAKE PLOTS

## A - schematic of sz distance to target
ax = axes['main-left-top'][0]
rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj - 0.06)
sch_path = '/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/schematic-targets-distance-to-sz.png'
img = mpimg.imread(sch_path)
axes['main-left-top'][0].imshow(img, interpolation='none')
axes['main-left-top'][0].axis('off')
axes['main-left-top'][0].set_title('Distance to seizure boundary')

## B - Flu change during recruitment to seizure of target
ax = axes['main-left-bottom'][0]
rfv.add_label_axes(text="B", ax=ax, x_adjust=x_adj - 0.02)
plot__targets_sz_invasion_meantraces(fig=fig, ax=axes['main-left-bottom'][0])
axes['main-left-bottom'][0].set_title('Time to seizure recruitment')
# fig.show()

# %%B' - photostim responses relative to time to seizure recruitment
ax = axes['main-right-bottomhigh'][0]
rfv.add_label_axes(text="B'", ax=ax, x_adjust=x_adj + 0.03)
main_temporal.plot__responses_v_szinvtemporal_no_normalization(results=results_temporal, save_path_full=f'{SAVE_FIG}/responses_sz_temporal_binned_line_plot.png',
                                                               axes=(axes['main-right-bottomhigh'], axes['main-right-bottomlow']), fig=fig)

# %%
fig.show()


# %%
if save_fig:
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure5-RF.png")
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure5-RF.svg")
