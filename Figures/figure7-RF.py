"""
todo:

- make schematic for distance of nontargets to seizure analysis
- double check distances, etc. is everything in place - responses correct? distances correct?

"""


import sys

from Figures.figure7_plots import z_score_response_proximal_distal, influence_response_proximal_and_distal
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _utils_.rfv_funcs import make_fig_layout, show_test_figure_layout, add_label_axes

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets

import numpy as np

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults

import _alloptical_utils as Utils
from _utils_.alloptical_plotting import plot_settings

plot_settings()

SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis


# %% SETUP
## Set general plotting parameters
# rfv.set_fontsize(7)

test = 0
save_fig = True if not test else False
dpi = 100 if test else 300

np.random.seed(2)  # fix seed


# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'main-left': {'panel_shape': (1, 1),
                'bound': (0.13, 0.80, 0.25, 0.95)},
    'main-right': {'panel_shape': (2, 1),
                'bound': (0.40, 0.80, 0.90, 0.95),
             'wspace': 0.8}
}


fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)


# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% ADD PLOTS TO AXES  ##################################################################################################################

z_score_response_proximal_distal(fig=fig, ax=axes['main-left'][0], results=results)
axes['main-left'][0].set_title('')
axes['main-left'][0].set_ylim([-0.075, 0.25])

influence_response_proximal_and_distal(fig=fig, axs=axes['main-right'], results=results)
axes['main-right'][0].set_xticklabels([0, 200, 400], fontsize=10)
axes['main-right'][1].set_xticklabels([0, 200, 400], fontsize=10)
axes['main-right'][0].set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)
axes['main-right'][1].set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)


# %% ADD PANEL LABELS

x_adj = 0.11

ax=axes['main-left'][0]
rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj)

ax=axes['main-right'][0]
rfv.add_label_axes(text='B', ax=ax, x_adjust=x_adj + 0.02)

ax=axes['main-right'][1]
rfv.add_label_axes(text="B'", ax=ax, x_adjust=x_adj + 0.02)

# %%

if save_fig and dpi > 250:
    Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/figure7-RF.png')
    Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/figure7-RF.svg')

fig.show()


