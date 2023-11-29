"""
Supplemental Figure 3 - Local circuit excitability in baseline and interictal states.

"""

# %%
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import _utils_.alloptical_plotting
import alloptical_plotting
from _exp_metainfo_.exp_metainfo import ExpMetainfo, baseline_color, interictal_color

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

# plot_settings()
SAVE_FOLDER = f'/home/pshah/Documents/figures/alloptical_seizures_draft/'

# %% SETUP
## Set general plotting parameters
fs = ExpMetainfo.figures.fontsize['extraplot']
rfv.set_fontsize(fs)

# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, b, r, t

# layout = {
#     'A': {'panel_shape': (1, 1),
#              'bound': (0.10, 0.80, 0.25, 0.95)},
#     'B': {'panel_shape': (1, 2),
#                     'bound': (0.37, 0.70, 0.52, 0.95)},
#     'C': {'panel_shape': (2, 1),
#           'bound': (0.63, 0.70, 0.84, 0.80),
#           'wspace': 1.3},
#     'D': {'panel_shape': (1, 1),
#           'bound': (0.10, 0.55, 0.30, 0.70),
#           'wspace': 0.8}
# }

layout = {
    'A': {'panel_shape': (1, 1),
             'bound': (0.10, 0.82, 0.25, 0.95)},
    'B': {'panel_shape': (2, 1),
          'bound': (0.36, 0.82, 0.70, 0.95), 'wspace': 0.1},
    'C': {'panel_shape': (2, 1),
          'bound': (0.78, 0.82, 0.95, 0.95),
          'wspace': 0.6},
    'D': {'panel_shape': (1, 1),
          'bound': (0.10, 0.59, 0.25, 0.72),
          'wspace': 0.8}
}


test = 1
save_fig = True if not test > 0 else False
dpi = 150 if test > 0 else 300
fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=dpi)
rfv.show_test_figure_layout(fig, axes=axes, show=True) if test == 2 else None  # test what layout looks like quickly, but can also skip and moveon to plotting data.




# %% MAKE PLOTS
x_adj = 0.09

# legend for B and C plots

# Create the legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

legend_elements = [Line2D([0], [0], marker='o', color=baseline_color, label='Baseline', lw=2,
                          markerfacecolor=baseline_color, markersize=7, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color=interictal_color, label='Interictal', lw=2,
                          markerfacecolor=interictal_color, markersize=7, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='gray', label='Artificial stimulation', lw=2,
                          markerfacecolor='white', markersize=7, markeredgecolor='black'),
                   ]

bbox = np.array(axes['B'][0].get_position())
bbox = Bbox.from_extents(bbox[0, 1]-0.3, bbox[1, 1] - 0.5, bbox[1, 0], bbox[1, 1])
# bbox = Bbox.from_extents(bbox[1, 0], bbox[0, 0], bbox[0, 1] + 0.00, bbox[0, 1] - 0.02)
# ax2 = fig.add_subplot(position = bbox)
axlegend = fig.add_subplot()
axlegend.set_position(pos=bbox)
axlegend.legend(handles=legend_elements, loc='center')
axlegend.axis('off')
# fig.show()




# %% B and C) total z scored responses of targets vs. total z scored responses of nontargets - photostim vs. sham stim
axs = (axes['B'], axes['C'])
rfv.add_label_axes(text='B', ax=axs[0][0], x_adjust=x_adj + 0.02, y_adjust=0.00)
rfv.add_label_axes(text='C', ax=axs[1][0], x_adjust=x_adj - 0.01, y_adjust=0.00)

# main.collect__zscored_summed_activity_vs_targets_activity(results=results)

print(f"Number of trials measured - baseline - {len(results.summed_responses['baseline']['exp'])} photostim trials")
print(f"Number of trials measured - interictal - {len(results.summed_responses['interictal']['exp'])} photostim trials")


# B + B') ratio of regression lines between baseline and interictal stims - photostim + fakestim
main.plot__summed_activity_vs_targets_activity(results=results, SAVE_FOLDER=SAVE_FOLDER, fig=fig, axs=axs)



axs[0][0].text(x=8, y = -29.5, s=f'Total targets\n' + r'response ($\it{z}$-scored)', fontsize=fs, clip_on=False, ha='center')

# axs[1][0].text(x=10, y=2.0, s=f'Photostimulation/Artificial\nratio', ha='center', va='center', fontsize=10)



# %% D) influence measurements - baseline + interictal across distance to targets
# PLOTTING of average responses +/- sem across distance to targets bins - baseline + interictal
axs = axes['D']
rfv.add_label_axes(text='D', ax=axs[0], x_adjust=x_adj - 0.00)

distance_lims = [19, 600]

measurement = 'new influence response'  # refers to the influence measurement where expected activity is inferred based on the overall population

ax = axs[0]
ax.axhline(y=0, ls='--', color='gray', lw=1)
ax.axvline(x=20, ls='--', color='gray', lw=1)

# print total number of cells measured:
print(f"Total number of cells measured - baseline - {len(np.unique(pj.flattenOnce(results.binned_distance_vs_responses[measurement]['cells measured'])))} nontargets")
print(f"Total number of cells measured - interictal - {len(np.unique(pj.flattenOnce(results.binned_distance_vs_responses_interictal[measurement]['cells measured'])))} nontargets")


# BASELINE- distances vs. responses
distances = results.binned_distance_vs_responses[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.3, color='royalblue')
ax.plot(distances, avg_binned_responses, lw=1.5, color=baseline_color, label='baseline')


# INTERICTAL - binned distances vs responses
distances = results.binned_distance_vs_responses_interictal[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.3, color='mediumseagreen')
ax.plot(distances, avg_binned_responses, lw=1.5, color=interictal_color, label='interictal')

# ax.set_title(f"{measurement}", wrap=True)
ax.set_xlim([0, 400])
ax.set_ylim([-0.175, 0.25])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='Distance to target ($\mu$$\it{m}$)', y_label='Photostimulation\ninfluence')
# ax.legend(loc='lower right')
# fig.suptitle('BASELINE + INTERICTAL')
# fig.tight_layout()
# fig.show()

ax.set_xticks([0, 200, 400], [0, 200, 400], fontsize=fs)
ax.set_yticks([0, 0.2], [0,0.2], fontsize=fs)

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
ax = axes['A'][0]
rfv.add_label_axes(text='A', ax=ax, x_adjust=x_adj, y_adjust=0.00)

baseline_responses = results.baseline_responses.iloc[results.pre4ap_idxs]
# num occurrences at each distance - split by trial types
distances = []
for exp in np.unique(baseline_responses['expID']):
    _distances = list(baseline_responses[baseline_responses['expID'] == exp]['distance target'])  # need to update to only count unique target to nontarget distances! - there's repeat for all the stims for the same measurement i think.....
    distances.append(np.unique(_distances))
ax.hist(distances, 40, density=False, histtype='bar', stacked=True)
# ax.set_yticks([0, 20000, 40000, 60000, 80000], ['0', '2', '4', '6', '8'], fontsize=fs)
ax.set_xticks([0, 300,  600], [0, 300, 600], fontsize=fs)
# ax.text(x=-50, y=82000, s='x 10$^{4}$', fontsize=fs, clip_on=False)
ax.set_ylabel('Number of\nnon targets', fontsize=fs)

# ax.ticklabel_format(axis='y', style='scientific', useMathText=False)
# ax.set_title('number of measurements by individual experiments')
ax.set_xlabel('Distance to target ($\mu$$\it{m}$)', fontsize=fs)

# %%
if save_fig and dpi > 250:
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure5-RF.png")
    _utils_.alloptical_plotting.save_figure(fig=fig, save_path_full=f"{SAVE_FOLDER}/figure5-RF.pdf")

fig.show()


