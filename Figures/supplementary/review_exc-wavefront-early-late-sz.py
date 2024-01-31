"""
Additional analysis in response to reviewer for testing excitability measurement vs. wavefront in early vs. late seizures

"""

import sys
import numpy as np
from funcsforprajay.funcs import flattenOnce

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main']); import rep_fig_vis as rfv

fig_title = f'review_exc-wavefront-early-late-sz'
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/4fig/'
fig_items = f'/home/pshah/Documents/figures/alloptical_seizures_draft/figure-items/'

from _utils_.alloptical_plotting import save_figure
from _exp_metainfo_.exp_metainfo import ExpMetainfo

from _analysis_._ClassTargetsSzInvasionSpatial_codereview import TargetsSzInvasionSpatial_codereview, \
    TargetsSzInvasionSpatialResults_codereview

main_spatial = TargetsSzInvasionSpatial_codereview
results_spatial: TargetsSzInvasionSpatialResults_codereview = TargetsSzInvasionSpatialResults_codereview.load()

from _utils_.alloptical_plotting import plot_settings

plot_settings()
np.random.seed(2)  # fix seed

fs = ExpMetainfo.figures.fontsize['extraplot']
rfv.set_fontsize(fs)

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis

# %% SETUP
# MAKING LAYOUT:

layout = {
    'AB': {'panel_shape': (2, 1),
                   'bound': (0.15, 0.77, 0.80, 0.92),
                   'wspace': 0.3},
}

test = 0
save_fig = True if not test > 0 else False
dpi = 100 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# %% ADD PLOTS TO AXES  ##################################################################################################################
ax_a = axes['AB'][0]


results_obj =  results_spatial.rolling_binned__distance_vs_photostimresponses_firstsz

# distances_bins = results.rolling_binned__distance_vs_photostimresponses['distance_bins']
distances = results_obj['distance_bins']
avg_responses = results_obj['avg_photostim_response_in_bin']
conf_int = results_obj['95conf_int']
num2 = results_obj['num_points_in_bin']

conf_int_distances = flattenOnce([[distances[i], distances[i + 1]] for i in range(len(distances) - 1)])
conf_int_values_neg = flattenOnce([[val, val] for val in conf_int[1:, 0]])
conf_int_values_pos = flattenOnce([[val, val] for val in conf_int[1:, 1]])

#### MAKE PLOT

# ax.plot(distances[:-1], avg_responses, c='cornflowerblue', zorder=1)
ax = ax_a
# ax.fill_between(x=(distances-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
# ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='#e7bcbc', zorder=2, alpha=1)
ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
                zorder=2, alpha=1)
ax.step(distances, avg_responses, c='cornflowerblue', zorder=3)
# ax.scatter(distances[:-1], avg_responses, c='orange', zorder=4)
# ax.set_title(
#     f'photostim responses vs. distance to sz wavefront (binned every {results.rolling_binned__distance_vs_photostimresponses["bin_width_um"]}um)',
#     wrap=True)
# ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)')
ax.set_xticks([0, 100, 200, 300, 400], [0, 100, 200, 300, 400], fontsize=10)
ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)', fontsize=10)
ax.set_ylim([-2, 3])
ax.set_xlim([0, 300])
# ax.set_yticks([-1, 0, 1, 2], [-1, 0, 1, 2], fontsize=10)
# ax.set_ylabel(f'{rfv.italic("Z")}-score\n(to baseline)',fontsize=10)
ax.set_ylabel(f'Photostimulation responses \n (z-score)', fontsize=10)
ax.axhline(0, ls='--', lw=1, color='black', zorder=0)

ax.set_title(f'Early seizure', fontsize=10)
ax.margins(0)

# %%
ax_b = axes['AB'][1]


results_obj =  results_spatial.rolling_binned__distance_vs_photostimresponses_lastsz

# distances_bins = results.rolling_binned__distance_vs_photostimresponses['distance_bins']
distances = results_obj['distance_bins']
avg_responses = results_obj['avg_photostim_response_in_bin']
conf_int = results_obj['95conf_int']
num2 = results_obj['num_points_in_bin']

conf_int_distances = flattenOnce([[distances[i], distances[i + 1]] for i in range(len(distances) - 1)])
conf_int_values_neg = flattenOnce([[val, val] for val in conf_int[1:, 0]])
conf_int_values_pos = flattenOnce([[val, val] for val in conf_int[1:, 1]])

#### MAKE PLOT

# ax.plot(distances[:-1], avg_responses, c='cornflowerblue', zorder=1)
ax = ax_b
# ax.fill_between(x=(distances-0)[:-1], y1=conf_int[:-1, 0], y2=conf_int[:-1, 1], color='lightgray', zorder=0)
# ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='#e7bcbc', zorder=2, alpha=1)
ax.fill_between(x=conf_int_distances, y1=conf_int_values_neg, y2=conf_int_values_pos, color='lightgray',
                zorder=2, alpha=1)
ax.step(distances, avg_responses, c='forestgreen', zorder=3)
# ax.scatter(distances[:-1], avg_responses, c='orange', zorder=4)
# ax.set_title(
#     f'photostim responses vs. distance to sz wavefront (binned every {results.rolling_binned__distance_vs_photostimresponses["bin_width_um"]}um)',
#     wrap=True)
# ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)')
ax.set_xticks([0, 100, 200, 300, 400], [0, 100, 200, 300, 400], fontsize=10)
ax.set_ylim([-2, 3])
ax.set_xlim([0, 300])
# ax.set_yticks([-1, 0, 1, 2], [-1, 0, 1, 2], fontsize=10)
# ax.set_ylabel(f'{rfv.italic("Z")}-score\n(to baseline)',fontsize=10)
# ax.set_ylabel(f'Photostimulation responses \n (z-score)', fontsize=10)
ax.set_xlabel(r'Distance to seizure wavefront ($\mu$$\it{m}$)', fontsize=10)
ax.axhline(0, ls='--', lw=1, color='black', zorder=0)

ax.set_title(f'Late seizure', fontsize=10)
ax.margins(0)

# %%
if save_fig and dpi > 250:
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.png")
    save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/{fig_title}.pdf")

fig.show()

