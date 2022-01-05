# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import os
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')

import alloptical_utils_pj as aoutils

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'

force_remake = False

if not os.path.exists(results_object_path) or force_remake:
    allopticalResults = aoutils.AllOpticalResults(save_path=results_object_path)
    # make a metainfo attribute to store all metainfo types of info for all experiments/trials
    allopticalResults.metainfo = allopticalResults.slmtargets_stim_responses.loc[:, ['prep_trial', 'date', 'exptype']]

allopticalResults = aoutils.import_resultsobj(results_object_path)

# %% 1) lists of trials to analyse for run_pre4ap_trials and run_post4ap_trials trials within experiments,
# note that the commented out trials are used to skip running processing code temporarily

allopticalResults.pre_4ap_trials = [
    ['RL108 t-009'],
    # ['RL108 t-010'],
    # ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],
    # ['RL109 t-014'],
    # ['PS04 t-012', 'PS04 t-014',  # - not sure what's wrong with PS04, but the photostim and Flu are falling out of sync .21/10/09
    #  'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    # ['PS07 t-009'],
    # ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],  # matching run_post4ap_trials trial cannot be analysed
    ['PS06 t-011'],
    # ['PS06 t-012'],  # matching run_post4ap_trials trial cannot be analysed
    # ['PS11 t-007'],
    ['PS11 t-010'],
    # ['PS17 t-005'],
    # ['PS17 t-006', 'PS17 t-007'],
    # ['PS18 t-006']
]

allopticalResults.post_4ap_trials = [
    ['RL108 t-013'],
    # ['RL108 t-011'],
    # ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    #['RL109 t-016'],  'RL109 t-017'], -- need to do sz boundary classifying processing
    # ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    # ['PS07 t-017'],
    # ['PS06 t-014', 'PS06 t-015'], - missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    ['PS06 t-013'],
    # ['PS06 t-016'], - no seizures, missing seizure_lfp_onsets (no paired measurements mat file for trial .21/10/09)
    # ['PS11 t-016'],
    ['PS11 t-011'],
    # ['PS17 t-011'],
    # ['PS17 t-009'],
    # ['PS18 t-008']
]

assert len(allopticalResults.pre_4ap_trials) == len(allopticalResults.post_4ap_trials), (
f"# of pre trials: {len(allopticalResults.pre_4ap_trials)}",
f"# of post trials: {len(allopticalResults.post_4ap_trials)}")

allopticalResults.trial_maps = {'pre': {}, 'post': {}}
allopticalResults.trial_maps['pre'] = {
    'a': ['RL108 t-009'],
    # 'b': ['RL108 t-010'],
    # 'c': ['RL109 t-007'],
    'd': ['RL109 t-008'],
    'e': ['RL109 t-013'],
    'f': ['RL109 t-014'],
    # 'g': ['PS04 t-012',  # 'PS04 t-014',  # - temp just until PS04 gets reprocessed
    #       'PS04 t-017'],
    'h': ['PS05 t-010'],
    'i': ['PS07 t-007'],
    # 'j': ['PS07 t-009'],
    # 'k': ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    'l': ['PS06 t-011'],
    # 'm': ['PS06 t-012'],  # - t-016 missing sz lfp onsets
    # 'n': ['PS11 t-007'],
    'o': ['PS11 t-010'],
    # 'p': ['PS17 t-005'],
    # 'q': ['PS17 t-006', 'PS17 t-007'],
    # 'r': ['PS18 t-006']
}

allopticalResults.trial_maps['post'] = {
    'a': ['RL108 t-013'],
    # 'b': ['RL108 t-011'], -- need to redo sz boundary classifying processing
    # 'c': ['RL109 t-020'], -- need to redo sz boundary classifying processing
    'd': ['RL109 t-021'],
    'e': ['RL109 t-018'],
    'f': ['RL109 t-016'],  # 'RL109 t-017'], -- need to do sz boundary classifying processing
    # 'g': ['PS04 t-018'],  -- need to redo sz boundary classifying processing
    'h': ['PS05 t-012'],
    'i': ['PS07 t-011'],
    # 'j': ['PS07 t-017'],  # - need to do sz boundary classifying processing
    # 'k': ['PS06 t-014', 'PS06 t-015'],  # - missing seizure_lfp_onsets
    'l': ['PS06 t-013'],
    # 'm': ['PS06 t-016'],  # - missing seizure_lfp_onsets - LFP signal not clear, but there is seizures on avg Flu trace
    # 'n': ['PS11 t-016'],
    'o': ['PS11 t-011'],
    # 'p': ['PS17 t-011'],
    # 'q': ['PS17 t-009'],
    # 'r': ['PS18 t-008']
}

assert len(allopticalResults.trial_maps['pre'].keys()) == len(allopticalResults.trial_maps['post'].keys()), (
f"# of pre trials: {len(allopticalResults.trial_maps['pre'].keys())}",
f"# of post trials: {len(allopticalResults.trial_maps['post'].keys())}")

allopticalResults.save()
