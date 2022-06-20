"""
TODO:
- add bar plots + stats comparing correlation of targets within exp to across experiment
    - for baseline, and for interictal
- add bar plot + stats comparing avg targets correlation within baseline to within interictal
- add suppl figure showing normal distribution of targets responses (normalized across x-axis to avg dFF for each target)

"""


# %%
import os

import numpy as np
from funcsforprajay.funcs import flattenOnce
from funcsforprajay.plotting.plotting import plot_bar_with_points

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets, plot__avg_photostim_dff_allexps
from _analysis_._ClassPhotostimResponseQuantificationSLMtargets import PhotostimResponsesSLMtargetsResults, \
    PhotostimResponsesQuantificationSLMtargets
from _utils_.alloptical_plotting import plot_settings
from alloptical_utils_pj import save_figure

main = PhotostimAnalysisSlmTargets
RESULTS = PhotostimResponsesSLMtargetsResults.load()

import sys

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import matplotlib.image as mpimg
import rep_fig_vis as rfv

## Set general plotting parameters
plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

rfv.set_fontsize(7)



# %%
## Set parameters
n_cat = 2
n_misc_rows = 2
n_misc = 5
colour_list = ['#101820', '#1b362c', '#2f553d', '#4f7553', '#79936f', '#aeae92']
colours_misc_dict = {xx: colour_list[xx] for xx in range(len(colour_list))}

save_fig = True

np.random.seed(2)  # fix seed


# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound should be = l, b, r, t

layout = {
    'main-top': {'panel_shape': (10, 2),
                'bound': (0.05, 0.75, 0.95, 0.95),
                 'hspace': 0.2},
    'main-middle-left': {'panel_shape': (1, 1),
                 'bound': (0.05, 0.55, 0.22, 0.67)},
    'main-middle-middle': {'panel_shape': (2, 1),
                 'bound': (0.29, 0.55, 0.69, 0.67),
                          'wspace': 0.2},
    'main-middle-right': {'panel_shape': (1, 1),
                 'bound': (0.80, 0.55, 0.9, 0.67)},
    'main-bottom-top': {'panel_shape': (2, 1),
                 'bound': (0.05, 0.28, 0.50, 0.45),
                          'wspace': 0.1},
    'main-bottom-low': {'panel_shape': (2, 1),
                 'bound': (0.04, 0.10, 0.52, 0.255),
                          'wspace': 0.01},
    'main-bottom-right': {'panel_shape': (2, 1),
                 'bound': (0.62, 0.28, 0.87, 0.43),
                          'wspace': 0.2},
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=50)

for ax in flattenOnce(axes['main-top']):
    rfv.naked(ax)
    ax.axis('off')
# rfv.naked(axes['main-bottom-left'][0])

# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% add plots to axes

axs=axes['main-bottom-right']  #: correlation matrixes
main.correlation_magnitude_exps(fig=fig, axs=axs)
rfv.add_label_axes(s='G', ax=axs[0], y_adjust=0.1, x_adjust = 0.09)
# fig.show()

axs=(axes['main-bottom-top'], axes['main-bottom-low'])  #: correlation matrixes
main.correlation_matrix_all_targets(fig=fig, axs=axs)
rfv.add_label_axes(s='F', ax=axs[0][0], y_adjust=0)
rfv.add_label_axes(s="F'", ax=axs[1][0], y_adjust=0)


axs=axes['main-middle-middle']  #: CV quantification bar plot
main.plot__mean_response_vs_variability(fig, axs = axs, rerun=0)
rfv.add_label_axes(s='E', ax=axs[0])


ax=axes['main-middle-left'][0]  #: CV quantification bar plot
main.plot__variability(fig=fig, ax=ax)
rfv.add_label_axes(s='B', ax=ax)


ax=axes['main-top'][0, 0]  #: CV representative examples
main.plot_variability_photostim_traces_by_targets(axs=axes['main-top'], fig=fig)
rfv.add_label_axes(s='A', ax=ax)


ax=axes['main-middle-right'][0]  #: interictal split - z scores
plot_bar_with_points(data=[
    RESULTS.interictal_responses['preictal_responses'],
    RESULTS.interictal_responses['very_interictal_responses'],
    RESULTS.interictal_responses['postictal_responses']
                                     ],
    bar=False, title='Interictal phases', x_tick_labels=['pre', 'mid.', 'post'],
    colors=['lightseagreen', 'gold', 'lightcoral'], figsize=(4, 4), y_label=RESULTS.interictal_responses['data_label'], show=False, ylims=[-0.5, 0.5], alpha=1, fig=fig, ax=ax, s=25)
rfv.add_label_axes(s='D', ax=ax, x_adjust=0.1)

fig.show()

if save_fig:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure4-RF.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure4-RF.svg")





