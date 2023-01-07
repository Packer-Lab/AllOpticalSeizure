"""
TODO:
[ ] combining the old fig 6 and 7 into one figure

- make schematic for distance of nontargets to seizure analysis
    fig legend for this: Non-targeted neurons were split into proximal (pink, < 100um from seizure wavefront) and distal (gold, > 200um from) to seizure wavefront groups for all trials to analyse their photostimulation response during ictal states.

- double check distances, etc. is everything in place - responses correct? distances correct?


STATS:
- change nontargets response magnitude stats to one way anova



suppl figure: write up RF code
- seizure boundary classification
- responses vs. pre-stim Flu targets annulus, across all conditions
    + stats tests on the averages? or can tests be carried out directly on the 2-D density plots?


"""

import sys

from _analysis_.nontargets_analysis._ClassNonTargetsSzInvasionSpatial import NonTargetsSzInvasionSpatialResults, \
    NonTargetsSzInvasionSpatial
from _exp_metainfo_.exp_metainfo import ExpMetainfo

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv
import alloptical_utils_pj as Utils

import numpy as np
import matplotlib.image as mpimg

from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview, \
    TargetsSzInvasionSpatialResults_codereview

SAVE_FIG = "/home/pshah/Documents/figures/alloptical-photostim-responses-sz-distance/"

main_spatial = TargetsSzInvasionSpatial_codereview
results_spatial = TargetsSzInvasionSpatialResults_codereview.load()

from _utils_.alloptical_plotting import plot_settings

plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

from _utils_.nontargets_responses_ictal_plots import z_score_response_proximal_distal, influence_response_proximal_and_distal

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis

# %% SETUP
## Set general plotting parameters

fs = ExpMetainfo.figures.fontsize['extraplot']
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
    'main-right': {'panel_shape': (2, 1),
                   'bound': (0.40, 0.77, 0.95, 0.92),
                   'wspace': 0.2},
    'main-left-mid': {'panel_shape': (1, 1),
                      'bound': (0.11, 0.50, 0.23, 0.65)},
    'main-right-mid': {'panel_shape': (2, 1),
                       'bound': (0.40, 0.50, 0.95, 0.65),
                       'wspace': 0.2}
}


test = 0
save_fig = True if not test > 0 else False
dpi = 150 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# ADD PLOTS TO AXES  ##################################################################################################################

# %% C - photostim responses of nontargets classed to interictal or seizure distance
ax_c = axes['main-left-mid'][0]
z_score_response_proximal_distal(fig=fig, ax=ax_c, results=results)
ax_c.set_title(f'Response magnitude\nNon-targets', fontsize=10)
ax_c.set_ylabel(f'{rfv.italic("Z")}-score\n(to baseline)', fontsize=10)
ax_c.set_ylim([-0.075, 0.25])


# %% A - schematic of sz distance to target
ax_a = axes['main-left-top'][0]
# rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj - 0.06)
sch_path = '/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/schematic-targets-distance-to-sz.png'
img = mpimg.imread(sch_path)
ax_a.imshow(img, interpolation='none')
ax_a.axis('off')
# axes['main-left-top'][0].set_title('Distance to seizure boundary')


# %% B - photostim responses relative to distance to seizure
ax_b = axes['main-right'][0]
ax_b1 = axes['main-right'][1]

# rfv.add_label_axes(text="A'", ax=ax, x_adjust=x_adj + 0.03)
main_spatial.collect__binned__distance_v_responses(results=results_spatial, rerun=0)

# ROLLING BINS:
main_spatial.collect__binned__distance_v_responses_rolling_bins(results=results_spatial, rerun=0)

results_spatial = TargetsSzInvasionSpatialResults_codereview.load()
main_spatial.plot__responses_v_distance_no_normalization_rolling_bins(results=results_spatial, axes=[ax_b, ], fig=fig)



# adding neuropil signal
results = NonTargetsSzInvasionSpatialResults.load()
NonTargetsSzInvasionSpatial.plot__firingrate_v_distance_no_normalization_rolling_bins(results=results, axes=ax_b1, fig=fig)





# %% D - photostim influence of nontargets classed to seizure distance
ax_d = axes['main-right-mid']
influence_response_proximal_and_distal(fig=fig, axs=ax_d, results=results)
# ax_d[0].set_xticks([0, 200, 400], fontsize=10)
# ax_d[1].set_xticks([0, 200, 400], fontsize=10)
ax_d[0].set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)
ax_d[1].set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)


# %% ADD PANEL LABELS

rfv.add_label_axes(text='A', ax=ax_a, x_adjust=0.03)

x_adj = 0.11
rfv.add_label_axes(text='B', ax=ax_b, x_adjust=x_adj)
# rfv.add_label_axes(text="B'", ax=ax_b1, x_adjust=x_adj + 0.02)

rfv.add_label_axes(text='C', ax=ax_c, x_adjust=x_adj - 0.02)

rfv.add_label_axes(text='D', ax=ax_d[0], x_adjust=x_adj)

# rfv.add_label_axes(text="D'", ax=ax_d[1], x_adjust=x_adj + 0.02)


# %%
if save_fig and dpi >= 250:
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6combo-RF.png")
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6combo-RF.pdf")

fig.show()




