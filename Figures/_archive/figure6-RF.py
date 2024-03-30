"""
TODO:

[x] add schematic of seizure distance to target

STATS:
[x] repsonses vs. time to invasion


suppl figure: write up RF code
- seizure boundary classification
- responses vs. pre-stim Flu targets annulus, across all conditions
    + stats tests on the averages? or can tests be carried out directly on the 2-D density plots?

"""

# %%
import sys

import _utils_.alloptical_plotting
from _analysis_._ClassTargetsSzInvasionTemporal import TargetsSzInvasionTemporal, TargetsSzInvasionTemporalResults
from _analysis_.nontargets_analysis._ClassNonTargetsSzInvasionSpatial import NonTargetsSzInvasionSpatialResults, \
    NonTargetsSzInvasionSpatial

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import rep_fig_vis as rfv

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

fs = 10
rfv.set_fontsize(fs)

test = 0
save_fig = True if not test else False
dpi = 100 if test else 300

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'main-left-top': {'panel_shape': (1, 1),
                      'bound': (0.05, 0.77, 0.27, 0.92)},
    # 'main-left-bottom': {'panel_shape': (1, 1),
    #                      'bound': (0.08, 0.42, 0.26, 0.57)},
    'main-right': {'panel_shape': (2, 1),
                           'bound': (0.40, 0.77, 0.95, 0.92),
                           'wspace': 0.2},
    # 'main-right-tophigh': {'panel_shape': (1, 1, 'twinx'),
    #                        'bound': (0.40, 0.75, 0.90, 0.95),
    #                        'wspace': 0.8},
    # 'main-right-toplow': {'panel_shape': (1, 1),
    #                       'bound': (0.40, 0.66, 0.90, 0.71),
    #                       'wspace': 0.8},
    # 'main-right-bottomhigh': {'panel_shape': (1, 1),
    #                           'bound': (0.40, 0.40, 0.90, 0.60),
    #                           'wspace': 0.8},
    # 'main-right-bottomlow': {'panel_shape': (1, 1),
    #                          'bound': (0.40, 0.31, 0.90, 0.34),
    #                          'wspace': 0.8},
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)

# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

x_adj = 0.09


# %% A - schematic of sz distance to target
ax = axes['main-left-top'][0]
rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj - 0.06)
sch_path = '/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/schematic-targets-distance-to-sz.png'
img = mpimg.imread(sch_path)
axes['main-left-top'][0].imshow(img, interpolation='none')
axes['main-left-top'][0].axis('off')
# axes['main-left-top'][0].set_title('Distance to seizure boundary')


# %% A' - photostim responses relative to distance to seizure
ax = axes['main-right'][0]
ax2 = axes['main-right'][1]

# adding neuropil signal
results = NonTargetsSzInvasionSpatialResults.load()
NonTargetsSzInvasionSpatial.plot__firingrate_v_distance_no_normalization_rolling_bins(results=results, axes=ax2, fig=fig)


# rfv.add_label_axes(text="A'", ax=ax, x_adjust=x_adj + 0.03)
main_spatial.collect__binned__distance_v_responses(results=results_spatial, rerun=0)

# ROLLING BINS:
main_spatial.collect__binned__distance_v_responses_rolling_bins(results=results_spatial, rerun=0)

results_spatial = TargetsSzInvasionSpatialResults_codereview.load()
# main_spatial.plot__responses_v_distance_no_normalization(results=results_spatial, axes=(axes['main-right-tophigh'], axes['main-right-toplow']), fig=fig)
main_spatial.plot__responses_v_distance_no_normalization_rolling_bins(results=results_spatial, axes=[ax, ], fig=fig)
# ax.text(x=50, y=2, s=f'{results_spatial.binned__distance_vs_photostimresponses["kruskal - binned responses"]}', fontsize=5)
# ax.text(x=50, y=1.95, s=f'{results_spatial.binned__distance_vs_photostimresponses["anova oneway - binned responses"]}', fontsize=5)


# %%
if save_fig and dpi >= 250:
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6-RF.png")
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6-RF.svg")
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6-RF.pdf")

fig.show()


