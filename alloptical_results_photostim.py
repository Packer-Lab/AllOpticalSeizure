#%% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj


# import onePstim superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %% ########## BAR PLOT showing average success rate of photostimulation

trial = 't-010'
animal_prep = 'PS05'
date = '2021-01-08'
# IMPORT pkl file containing data in form of expobj
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object

expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)



# plot across different groups
t009_pre_4ap_reliability = list(expobj.StimSuccessRate_SLMtargets.values())
# t011_post_4ap_reliabilty = list(expobj.StimSuccessRate_cells.values())  # reimport another expobj for post4ap trial
t013_post_4ap_reliabilty = list(expobj.StimSuccessRate_SLMtargets.values())  # reimport another expobj for post4ap trial
#
pj.plot_bar_with_points(data=[t009_pre_4ap_reliability, t013_post_4ap_reliabilty], xlims=[0.25, 0.3],
                        x_tick_labels=['pre-4ap', 'post-4ap'], colors=['green', 'deeppink'], y_label='% success stims.',
                        ylims=[0, 100], bar=False, title='success rate of stim. responses', expand_size_y=1.2, expand_size_x=1.2)

# %% adding slm targets responses to alloptical results superobject.slmtargets_stim_responses

animal_prep = 'PS07'
date = '2021-01-19'
# trial = 't-009'

pre4ap_trials = ['t-007', 't-008', 't-009']
post4ap_trials = ['t-011', 't-016', 't-017']

# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
#     date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
#
# expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

counter = allopticalResults.slmtargets_stim_responses.shape[0] + 1
# counter = 6

for trial in pre4ap_trials + post4ap_trials:
    print(counter)
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
        date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object

    expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

    # add trials info to experiment
    expobj.metainfo['pre4ap_trials'] = pre4ap_trials
    expobj.metainfo['post4ap_trials'] = post4ap_trials
    expobj.save()

    # save to results object:
    allopticalResults.slmtargets_stim_responses.loc[counter, 'prep_trial'] = '%s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'])
    allopticalResults.slmtargets_stim_responses.loc[counter, 'date'] = expobj.metainfo['date']
    allopticalResults.slmtargets_stim_responses.loc[counter, 'exptype'] = expobj.metainfo['exptype']
    if hasattr(expobj, 'stims_in_sz'):
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean([[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean([[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(list(expobj.outsz_StimSuccessRate_SLMtargets.values()))
    else:
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean([[np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(list(expobj.StimSuccessRate_SLMtargets.values()))

    allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dFF all targets)'] = np.nan
    counter += 1

allopticalResults.save()
allopticalResults.slmtargets_stim_responses


# %% comparing avg. response magnitudes for pre4ap and post4ap within same experiment prep.

pre_4ap_trials = [
    ['RL108 t-009'],
    ['RL108 t-010'],
    ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],
    ['RL109 t-014'],
    ['PS04 t-012', 'PS04 t-014', 'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    ['PS07 t-008'],
    ['PS07 t-009'],
    ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    ['PS06 t-011'],
    ['PS06 t-012'],
    ['PS11 t-007'],
    ['PS11 t-010'],
    ['PS17 t-005'],
    ['PS17 t-006', 'PS17 t-007'],
    ['PS18 t-006']
]

post_4ap_trials = [
    ['RL108 t-013'],
    ['RL108 t-011'],
    ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016', 'RL109 t-017'],
    ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    ['PS07 t-016'],
    ['PS07 t-017'],
    ['PS06 t-014', 'PS06 t-015'],
    ['PS06 t-013'],
    ['PS06 t-016'],
    ['PS11 t-016'],
    ['PS11 t-011'],
    ['PS17 t-011'],
    ['PS17 t-009'],
    ['PS18 t-008']
]


pre4ap_response_magnitude = []
for i in pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
                                         allopticalResults.slmtargets_stim_responses['prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))


post4ap_response_magnitude = []
for i in post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
                                         allopticalResults.slmtargets_stim_responses['prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    post4ap_response_magnitude.append(np.mean(x))

# %%


pj.plot_bar_with_points(data=[pre4ap_response_magnitude, post4ap_response_magnitude], paired=True, colors=['black', 'purple'], bar=False, expand_size_y=0.9, expand_size_x=0.5,
                     xlims=True)

