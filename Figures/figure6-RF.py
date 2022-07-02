"""
TODO:

[ ] add schematic of seizure distance to target

suppl figure: write up RF code
- responders analysis

"""

# %%
import sys

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import rep_fig_vis as rfv
import alloptical_utils_pj as Utils
import numpy as np

import funcsforprajay.funcs as pj

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults
from _utils_.alloptical_plotting import plot_settings


main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

# %% SETUP
## Set general plotting parameters
rfv.set_fontsize(7)

## Set parameters
save_fig = True

np.random.seed(2)  # fix seed

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

layout = {
    'left': {'panel_shape': (1, 1),
                      'bound': (0.10, 0.80, 0.30, 0.95)},
    'middle-left': {'panel_shape': (1, 2),
                         'bound': (0.37, 0.70, 0.52, 0.95)},
    'middle-right': {'panel_shape': (1, 2),
                           'bound': (0.62, 0.72, 0.67, 0.95),
                           'wspace': 0.8},
    'right': {'panel_shape': (1, 1),
                          'bound': (0.75, 0.70, 0.95, 0.85),
                          'wspace': 0.8}
}
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=300)

# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.


# %% MAKE PLOTS
x_adj = 0.09

# %% B) total z scored responses of targets vs. total z scored responses of nontargets - photostim vs. sham stim
axs = (axes['middle-left'], axes['middle-right'])
rfv.add_label_axes(text='B', ax=axs[0][0], x_adjust=x_adj - 0.04, y_adjust=0.00)

# main.collect__zscored_summed_activity_vs_targets_activity(results=results)

# B + B') ratio of regression lines between baseline and interictal stims - photostim + fakestim
main.plot__summed_activity_vs_targets_activity(results=results, SAVE_FOLDER=SAVE_FOLDER, fig=fig, axs=axs)


# %% legend for B and C plots

# Create the legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

legend_elements = [Line2D([0], [0], marker='o', color='royalblue', label='Baseline', lw=2,
                          markerfacecolor='royalblue', markersize=7, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='forestgreen', label='Interictal', lw=2,
                          markerfacecolor='forestgreen', markersize=7, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='gray', label='Artificial stimulation', lw=2,
                          markerfacecolor='white', markersize=7, markeredgecolor='black'),
                   ]

bbox = np.array(axes['right'][0].get_position())
bbox = Bbox.from_extents(bbox[0, 1] + 0.02, bbox[1, 1] + 0.15, bbox[1, 0], bbox[1, 1])
# bbox = Bbox.from_extents(bbox[1, 0], bbox[0, 0], bbox[0, 1] + 0.00, bbox[0, 1] - 0.02)
# ax2 = fig.add_subplot(position = bbox)
axlegend = fig.add_subplot()
axlegend.set_position(pos=bbox)
axlegend.legend(handles=legend_elements, loc='center')
axlegend.axis('off')
# fig.show()

# %% C) influence measurements - baseline + interictal across distance to targets
# PLOTTING of average responses +/- sem across distance to targets bins - baseline + interictal
axs = axes['right']
rfv.add_label_axes(text='C', ax=axs[0], x_adjust=x_adj - 0.02)

distance_lims = [19, 400]

measurement = 'new influence response'
ax = axs[0]
ax.axhline(y=0, ls='--', color='gray', lw=1.5)
ax.axvline(x=20, ls='--', color='gray', lw=1.5)

# BASELINE- distances vs. responses
distances = results.binned_distance_vs_responses[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.3, color='royalblue')
ax.plot(distances, avg_binned_responses, lw=1.5, color='royalblue', label='baseline')


# binned distances vs responses
distances = results.binned_distance_vs_responses_interictal[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.3, color='mediumseagreen')
ax.plot(distances, avg_binned_responses, lw=1.5, color='mediumseagreen', label='interictal')

ax.set_title(f"{measurement}", wrap=True)
ax.set_xlim([0, 400])
ax.set_ylim([-0.175, 0.25])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='Distance to target (um)', y_label=measurement)
# ax.legend(loc='lower right')
# fig.suptitle('BASELINE + INTERICTAL')
# fig.tight_layout()
# fig.show()

## TWO WAY ANOVA
# build longform table for TWO-WAY ANOVA - baseline vs. interictal

#
# distance_responses_df = pd.DataFrame({
#     'group': [],
#     'distance': [],
#     'response': []
# })
#
# distance_response = results.binned_distance_vs_responses['new influence response']['distance responses']
#
# # distance_responses_df = []
# for distance, responses in distance_response.items():
#     if distance_lims[0] < distance < distance_lims[1]:
#         for response in responses:
#             _df = pd.DataFrame({
#                 'group': 'baseline',
#                 'distance': distance,
#                 'response': response
#             },  index=[f"baseline_{distance}"])
#             distance_responses_df = pd.concat([distance_responses_df, _df])
#
#
# distance_response = results.binned_distance_vs_responses_interictal['new influence response']['distance responses']
#
# for distance, responses in distance_response.items():
#     if distance_lims[0] < distance < distance_lims[1]:
#         for response in responses:
#             _df = pd.DataFrame({
#                 'group': 'interictal',
#                 'distance': distance,
#                 'response': response
#             },  index=[f"interictal_{distance}"])
#             distance_responses_df = pd.concat([distance_responses_df, _df])
#
#
#
# # perform TWO WAY ANOVA
#
# model = ols('response ~ C(group) + C(distance) + C(group):C(distance)', data=distance_responses_df).fit()
# sm.stats.anova_lm(model, typ=2)


# %% A - num targets at each distance - split by individual experiments
ax = axes['left'][0]
rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj, y_adjust=0.00)

baseline_responses = results.baseline_responses.iloc[results.pre4ap_idxs]
# num occurrences at each distance - split by trial types
distances = []
for exp in np.unique(baseline_responses['expID']):
    _distances = list(baseline_responses[baseline_responses['expID'] == exp]['distance target'])
    distances.append(_distances)
ax.hist(distances, 40, density=False, histtype='bar', stacked=True)
# ax.set_title('number of measurements by individual experiments')
ax.set_xlabel('Distance to target (um)')


# %%
# fig.show()


# %%
if save_fig:
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6-RF.png")
    Utils.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure6-RF.svg")
