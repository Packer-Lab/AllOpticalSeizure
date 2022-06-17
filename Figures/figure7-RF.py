import sys

from Figures.figure7 import z_score_response_proximal_distal, influence_response_proximal_and_distal
from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _utils_.rfv_funcs import make_fig_layout, show_test_figure_layout, add_label_axes

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
import funcsforprajay.plotting as pplot
import funcsforprajay.funcs as pj

import numpy as np

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults

import _alloptical_utils as Utils
from _utils_.alloptical_plotting import plot_settings


SAVE_FOLDER = f'/home/pshah/mnt/qnap/Analysis/figure-items'

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis

plot_settings()

# %% SETUP
## Set general plotting parameters
# rfv.set_fontsize(7)

## Set parameters
n_cols = 3
n_rows = 1

save_fig = True

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
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=100)


# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% ADD PLOTS TO AXES  ##################################################################################################################

z_score_response_proximal_distal(fig=fig, ax=axes['main-left'][0], results=results)
axes['main-left'][0].set_title('')
influence_response_proximal_and_distal(fig=fig, axs=axes['main-right'], results=results)


# %% ADD PANEL LABELS

x_adj = 0.11

ax=axes['main-left'][0]
rfv.add_label_axes(s='A', ax=ax, x_adjust=x_adj)

ax=axes['main-right'][0]
rfv.add_label_axes(s='B', ax=ax, x_adjust=x_adj)

ax=axes['main-right'][1]
rfv.add_label_axes(s="B'", ax=ax, x_adjust=x_adj)


fig.show()


# %%

if save_fig:
    Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/figure7-RF.png')
    Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/figure7-RF.svg')


