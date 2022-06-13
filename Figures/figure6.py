import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols

import funcsforprajay.funcs as pj

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults
from _utils_.alloptical_plotting import plot_settings

plot_settings()

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

# SAVE_FOLDER = f'/home/pshah/mnt/qnap/Analysis/figure-items'
SAVE_FOLDER = f'/home/pshah/Documents/code/AllOpticalSeizure/figure-items'


# %% B) total z scored responses of targets vs. total z scored responses of nontargets - photostim vs. sham stim

# main.collect__zscored_summed_activity_vs_targets_activity(results=results)

# B + B') ratio of regression lines between baseline and interictal stims - photostim + fakestim
main.plot__summed_activity_vs_targets_activity(results=results, SAVE_FOLDER=SAVE_FOLDER)


# %% A) # num targets at each distance - split by individual experiments

fig, ax = plt.subplots(figsize = (5,5))

baseline_responses = results.baseline_responses.iloc[results.pre4ap_idxs]
# num occurrences at each distance - split by trial types
distances = []
for exp in np.unique(baseline_responses['expID']):
    _distances = list(baseline_responses[baseline_responses['expID'] == exp]['distance target'])
    distances.append(_distances)
ax.hist(distances, 40, density=False, histtype='bar', stacked=True)
ax.set_title('number of measurements by individual experiments')
ax.set_xlabel('distance to target (um)')
fig.tight_layout(pad=0.2)
fig.show()



# %% C) influence measurements - baseline + interictal across distance to targets

# build longform table for TWO-WAY ANOVA - baseline vs. interictal

distance_lims = [19, 400]

distance_responses_df = pd.DataFrame({
    'group': [],
    'distance': [],
    'response': []
})

distance_response = results.binned_distance_vs_responses['new influence response']['distance responses']

# distance_responses_df = []
for distance, responses in distance_response.items():
    if distance_lims[0] < distance < distance_lims[1]:
        for response in responses:
            _df = pd.DataFrame({
                'group': 'baseline',
                'distance': distance,
                'response': response
            },  index=[f"baseline_{distance}"])
            distance_responses_df = pd.concat([distance_responses_df, _df])


distance_response = results.binned_distance_vs_responses_interictal['new influence response']['distance responses']

for distance, responses in distance_response.items():
    if distance_lims[0] < distance < distance_lims[1]:
        for response in responses:
            _df = pd.DataFrame({
                'group': 'interictal',
                'distance': distance,
                'response': response
            },  index=[f"interictal_{distance}"])
            distance_responses_df = pd.concat([distance_responses_df, _df])



# perform TWO WAY ANOVA

model = ols('response ~ C(group) + C(distance) + C(group):C(distance)', data=distance_responses_df).fit()
sm.stats.anova_lm(model, typ=2)

# %% 5.2) PLOTTING of average responses +/- sem across distance to targets bins - baseline + interictal

measurement = 'new influence response'
fig, ax = plt.subplots(figsize = (4, 4), dpi=300)

ax.axhline(y=0, ls='--', color='gray', lw=1)
ax.axvline(x=20, ls='--', color='gray', lw=1)

# BASELINE- distances vs. responses
distances = results.binned_distance_vs_responses[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.1, color='royalblue')
ax.plot(distances, avg_binned_responses, lw=1, color='royalblue', label='baseline')


# binned distances vs responses
distances = results.binned_distance_vs_responses_interictal[measurement]['distances']
distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
distances = distances[distances_lim_idx]
avg_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['avg binned responses'][distances_lim_idx]
sem_binned_responses = results.binned_distance_vs_responses_interictal[measurement]['sem binned responses'][distances_lim_idx]
ax.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.1, color='mediumseagreen')
ax.plot(distances, avg_binned_responses, lw=1, color='mediumseagreen', label='interictal')

ax.set_title(f"{measurement}", wrap=True)
ax.set_xlim([0, 400])
ax.set_ylim([-0.175, 0.25])
pj.lineplot_frame_options(fig=fig, ax=ax, x_label='distance to target (um)', y_label=measurement)
ax.legend(loc='lower right')
fig.suptitle('BASELINE + INTERICTAL')
fig.tight_layout()
fig.show()

