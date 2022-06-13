import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
import funcsforprajay.plotting as pplot
import funcsforprajay.funcs as pj

import pandas as pd
import numpy as np

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults
from _main_.AllOpticalMain import alloptical

import statsmodels.api as sm
from statsmodels.formula.api import ols

import _alloptical_utils as Utils
from _utils_.alloptical_plotting import plot_settings

plot_settings()

SAVE_FOLDER = f'/home/pshah/mnt/qnap/Analysis/figure-items'

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis


# %% A) BAR PLOT COMPARING PHOTOSTIM RESPONSES OF DISTAL VS. PROXIMAL NONTARGETS (WITHIN 250UM OF TARGET)
measurement = 'z score response'

distance_response = results.binned_distance_vs_responses_distal[measurement]['distance responses']
# run stats analysis on limited distances: ONE-WAY ANOVA:
distal_responses = []
for distance, responses in distance_response.items():
    if distance_lims[0] < distance < distance_lims[1]:
        distal_responses.append(responses)


distance_response = results.binned_distance_vs_responses_proximal[measurement]['distance responses']
# run stats analysis on limited distances: ONE-WAY ANOVA:
proximal_responses = []
for distance, responses in distance_response.items():
    if distance_lims[0] < distance < distance_lims[1]:
        proximal_responses.append(responses)

# %% A) make bar plot

plot_settings()

fig, ax = pplot.plot_bar_with_points(data=[pj.flattenOnce(proximal_responses), pj.flattenOnce(distal_responses)], points=False,
                           paired=False, bar=True, colors=['#db5aac', '#dbd25a'], edgecolor='black', lw = 1.25,
                           x_tick_labels=['proximal', 'distal'], y_label='z score (to baseline)', shrink_text=1.3,
                           title='avg z score response', ylims=[0, 0.25], show=False)

fig.tight_layout(pad=0.5)
fig.show()
Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_ictal_zscore_proximal_distal_bar.png')


# %% A) RUNNING STATS COMPARING SIGNIFICANCE OF DIFFERENCE IN MEAN PHOTOSTIM RESPONSE: DISTAL VS. PROXIMAL - some sort of t - test? mann whitney U test?


# run stats: t test form of some sort:




# %% B + B') RUNNING STATS COMPARING SIGNIFICANCE OF DISTANCE, DISTAL + PROXIMAL

measurement = 'new influence response'
# measurement = 'z score response'


# run stats: one way ANOVA:
# distance_response = results.binned_distance_vs_responses_proximal['new influence response']['distance responses']
distance_response = results.binned_distance_vs_responses_distal[measurement]['distance responses']
# run stats analysis on limited distances: ONE-WAY ANOVA:
args = []
for distance, responses in distance_response.items():
    if distance_lims[0] < distance < distance_lims[1]:
        args.append(responses)
stats.f_oneway(*args)



# %% B) PLOTTING of average responses +/- sem across distance to targets bins - DISTAL

measurement = 'new influence response'

plot_settings()

fig, ax = plt.subplots(figsize = (4, 4), dpi=300)

ax.axhline(y=0, ls='--', color='gray', lw=1)
ax.axvline(x=20, ls='--', color='gray', lw=1)

# DISTAL- binned distances vs responses
distances = np.asarray(results.binned_distance_vs_responses_distal[measurement]['distances'])
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses_distal[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses_distal[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.2, color='#dbd25a')
ax.plot(distances, avg_binned_responses, lw=1, color='#dbd25a', label='distal')

# ax.legend(loc='lower right')
ax.set_title(f"{measurement}", wrap=True)
ax.set_xlim([0, 400])
ax.set_ylim([-0.3, 0.5])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='distance to target (um)', y_label=measurement)

fig.suptitle('DISTAL')
fig.tight_layout()
# fig.show()

Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_distance_stim_influence_distal.png')

# %% B') PLOTTING of average responses +/- sem across distance to targets bins - PROXIMAL
plot_settings()

fig, ax = plt.subplots(figsize = (4, 4), dpi=300)

ax.axhline(y=0, ls='--', color='gray', lw=1)
ax.axvline(x=20, ls='--', color='gray', lw=1)

# PROXIMAL- distances vs. responses
distances = np.asarray(results.binned_distance_vs_responses_proximal[measurement]['distances'])
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses_proximal[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses_proximal[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.2, color='#db5aac')
ax.plot(distances, avg_binned_responses, lw=1, color='#db5aac', label='proximal')

# ax.legend(loc='lower right')
ax.set_title(f"{measurement}", wrap=True)
ax.set_xlim([0, 400])
ax.set_ylim([-0.3, 0.5])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='distance to target (um)', y_label=measurement)

fig.suptitle('PROXIMAL')
fig.tight_layout()
# fig.show()
Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_distance_stim_influence_proximal.png')


