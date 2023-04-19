import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
import funcsforprajay.plotting as pplot
import funcsforprajay.funcs as pj

import pandas as pd
import numpy as np
import pingouin as pg

from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults
from _main_.AllOpticalMain import alloptical

import statsmodels.api as sm
from statsmodels.formula.api import ols

import _alloptical_utils as Utils
from _utils_.alloptical_plotting import plot_settings
import sys
sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])
import rep_fig_vis as rfv

plot_settings()

SAVE_FOLDER = f'/home/pshah/mnt/qnap/Analysis/figure-items'

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

distance_lims = [19, 400]  # limit of analysis


# %% plotting definitions

def z_score_response_proximal_distal(results=results, **kwargs):
    """BAR PLOT COMPARING PHOTOSTIM RESPONSES OF DISTAL VS. PROXIMAL NONTARGETS (WITHIN 250UM OF TARGET)"""
    measurement = 'z score response'  # z scored to baseline distribution

    distance_response = results.binned_distance_vs_responses_distal[measurement]['distance responses']

    distal_responses = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            distal_responses.append(responses)
    distal_responses = pj.flattenOnce(distal_responses)


    distance_response = results.binned_distance_vs_responses_proximal[measurement]['distance responses']

    proximal_responses = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            proximal_responses.append(responses)
    proximal_responses = pj.flattenOnce(proximal_responses)


    distance_response = results.binned_distance_vs_responses_interictal[measurement]['distance responses']
    # run stats analysis on limited distances: ONE-WAY ANOVA:
    interictal_responses = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            interictal_responses.append(responses)
    interictal_responses = pj.flattenOnce(interictal_responses)

    distance_response = results.binned_distance_vs_responses[measurement]['distance responses']
    # run stats analysis on limited distances: ONE-WAY ANOVA:
    baseline_responses = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            baseline_responses.append(responses)
    baseline_responses = pj.flattenOnce(baseline_responses)

    # A) RUNNING STATS COMPARING SIGNIFICANCE OF DIFFERENCE IN MEAN PHOTOSTIM RESPONSE: DISTAL VS. PROXIMAL - some sort of t - test? mann whitney U test?

    # create DataFrame to hold data
    data_nums = []
    num_interictal = len(interictal_responses)
    num_distal = len(distal_responses)
    num_proximal = len(proximal_responses)
    data_nums.extend(['interictal'] * num_interictal)
    data_nums.extend(['distal'] * num_distal)
    data_nums.extend(['proximal'] * num_proximal)

    # make dataframe from interictal, distal, proximal responses
    data = pd.DataFrame({'response': interictal_responses + distal_responses + proximal_responses, 'group': data_nums})

    # perform post-hoc pairwise Games-Howell test
    aov = pg.anova(data=data, dv='response', between='group', detailed=True)
    print(f'ANOVA (Non-targets responses): {aov}')
    posthoc = pg.pairwise_gameshowell(data, dv='response', between='group')
    print(f'Games-Howell post-hoc test (Non-targets responses): {posthoc}')

    # # run stats: t test:
    # proximal = pj.flattenOnce(proximal_responses)
    # distal = pj.flattenOnce(distal_responses)
    # print(f"\nP(ttest - proximal vs. distal responses): {stats.ttest_ind(proximal, distal)[1]:.2e}\n")

    # A) make bar plot
    #### new bar plot including interictal and baseline z score responses

    fig, ax = plt.subplots(figsize=[3, 3]) if 'fig' not in kwargs and 'ax' not in kwargs else (kwargs['fig'], kwargs['ax'])
    fig, ax = pplot.plot_bar_with_points(data=[interictal_responses, proximal_responses, distal_responses], points=False,
                               paired=False, bar=True, colors=['royalblue', '#db5aac', '#dbd25a'], edgecolor='black', lw = 0.8, capsize=3.5,
                               x_tick_labels=['Interictal', 'Proximal', 'Distal'], y_label='Response magnitude\n($\it{z}$-score)', fontsize=10,
                               title='avg z score response', show=False, fig=fig, ax=ax)
    #### // end

    # fig.tight_layout(pad=0.5)
    # fig.show()

    #### bar plot not including interictal and baseline z score responses - only proximal and distal responses
    # fig, ax = (kwargs['fig'], kwargs['ax']) if 'fig' in kwargs else (None, None)
    # fig, ax = pplot.plot_bar_with_points(data=[pj.flattenOnce(proximal_responses), pj.flattenOnce(distal_responses)], points=False,
    #                            paired=False, bar=True, colors=['#db5aac', '#dbd25a'], edgecolor='black', lw = 1.25,
    #                            x_tick_labels=['Proximal', 'Distal'], y_label='Response magnitude \n($\it{z}$-score)', shrink_text=1.3,
    #                            title='avg z score response', ylims=[0, 0.25], show=False, fig=fig, ax=ax, sig_compare_lines={'*****': [0, 1]})
    #### // end


    # fig.tight_layout(pad=0.5)
    # fig.show()
    # Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_ictal_zscore_proximal_distal_bar.png')

    # return fig, ax



def influence_response_proximal_and_distal(axs, fig, results=results):
    # B + B') RUNNING STATS COMPARING SIGNIFICANCE OF DISTANCE, DISTAL + PROXIMAL

    measurement = 'new influence response'
    # measurement = 'z score response'


    # run stats: one way ANOVA:
    # PROXIMAL
    distance_response = results.binned_distance_vs_responses_proximal[measurement]['distance responses']
    args = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            args.append(responses)
    print(f"NONTARGETS PHOTOSTIM INFLUENCE: p(f_oneway, proximal): {stats.f_oneway(*args)[1]}")


    # DISTAL
    distance_response = results.binned_distance_vs_responses_distal[measurement]['distance responses']
    # run stats analysis on limited distances: ONE-WAY ANOVA:
    args = []
    for distance, responses in distance_response.items():
        if distance_lims[0] < distance < distance_lims[1]:
            args.append(responses)
    print(f"NONTARGETS PHOTOSTIM INFLUENCE: p(f_oneway, distal): {stats.f_oneway(*args)[1]}")



    # B) PLOTTING of average responses +/- sem across distance to targets bins - DISTAL

    assert axs.shape == (2,), f"axes shape: {axs.shape} is incorrect, needs to be (2,)"

    measurement = 'new influence response'

    plot_settings()

    # fig, ax = plt.subplots(figsize = (4, 4), dpi=300)
    ax_d = axs[1]

    ax_d.axhline(y=0, ls='--', color='darkgray', lw=1, zorder=2)
    ax_d.axvline(x=20, ls='--', color='darkgray', lw=1, zorder=2)

    # DISTAL- binned distances vs responses
    distances = np.asarray(results.binned_distance_vs_responses_distal[measurement]['distances'])
    distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
    distances = distances[distances_lim_idx]
    avg_binned_responses = results.binned_distance_vs_responses_distal[measurement]['avg binned responses'][distances_lim_idx]
    sem_binned_responses = results.binned_distance_vs_responses_distal[measurement]['sem binned responses'][distances_lim_idx]
    # ax_d.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=0.2, color='#dbd25a', zorder=1)
    ax_d.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=1, color='#f7f6de', zorder=0)
    ax_d.plot(distances, avg_binned_responses, lw=1, color='#dbd25a', label='distal', zorder=3)

    # ax.legend(loc='lower right')
    ax_d.set_xlim([0, 400])
    ax_d.set_xticks([0, 100, 200, 300, 400], [0, 100, 200, 300, 400], fontsize=10)
    # ax_d.set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)
    ax_d.set_yticks([-0.2, 0, 0.2, 0.4], ['', '', '', ''], fontsize=10)
    # ax_d.set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)
    ax_d.set_ylim([-0.3, 0.5])
    ax_d.margins(0.02)
    # pj.lineplot_frame_options(fig=fig, ax=ax, x_label='Distance to target ' + '($\mu$$\it{m}$)', y_label='Photostimulation influence')


    ax_d.set_title(f'Distal\n' + r'(>200 $\mu$$\it{m}$ to seizure)', fontsize=10)
    # fig.tight_layout()
    # fig.show()

    # Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_distance_stim_influence_distal.png')

    # B') PLOTTING of average responses +/- sem across distance to targets bins - PROXIMAL
    plot_settings()

    # fig, ax = plt.subplots(figsize = (4, 4), dpi=300)
    ax_p = axs[0]

    ax_p.axhline(y=0, ls='--', color='darkgray', lw=1, zorder=2)
    ax_p.axvline(x=20, ls='--', color='darkgray', lw=1, zorder=2)

    # PROXIMAL- distances vs. responses
    distances = np.asarray(results.binned_distance_vs_responses_proximal[measurement]['distances'])
    distances_lim_idx = [idx for idx, distance in enumerate(distances) if distance_lims[0] < distance < distance_lims[1]]
    distances = distances[distances_lim_idx]
    avg_binned_responses = results.binned_distance_vs_responses_proximal[measurement]['avg binned responses'][distances_lim_idx]
    sem_binned_responses = results.binned_distance_vs_responses_proximal[measurement]['sem binned responses'][distances_lim_idx]
    ax_p.fill_between(x=list(distances), y1=list(avg_binned_responses + sem_binned_responses), y2=list(avg_binned_responses - sem_binned_responses), alpha=1, color='#f7deee', zorder=0)
    ax_p.plot(distances, avg_binned_responses, lw=1, color='#db5aac', label='proximal', zorder=3)

    # ax.legend(loc='lower right')
    ax_p.set_title(f"{measurement}", wrap=True)
    ax_p.set_xlim([0, 400])
    ax_p.set_xticks([0, 100, 200, 300, 400], [0, 100, 200, 300, 400], fontsize=10)
    ax_p.set_yticks([-0.2, 0, 0.2, 0.4], [-0.2, 0, 0.2, 0.4], fontsize=10)
    ax_p.set_ylim([-0.3, 0.5])
    ax_p.set_ylabel(f'Photostimulation\ninfluence')
    ax_p.margins(0.02)
    # pj.lineplot_frame_options(fig=fig, ax=ax, x_label='Distance to target ' + '($\mu$$\it{m}$)', y_label='')


    ax_p.set_title(f'Proximal\n' + r'(<100 $\mu$$\it{m}$ to seizure)', fontsize=10)

    ax_p.text(x=230, y=-0.6, s=r'Distance to nearest target ($\mu$$\it{m}$)', fontsize=10)
    # fig.tight_layout()
    # fig.show()
    # Utils.save_figure(fig=fig, save_path_full=f'{SAVE_FOLDER}/nontargets_distance_stim_influence_proximal.png')

    # return fig, axs


if __name__ == '__main__':
    z_score_response_proximal_distal()

