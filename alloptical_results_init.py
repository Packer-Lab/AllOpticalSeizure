# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import os
import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')

import alloptical_utils_pj as aoutils

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'

force_redo = False

if not os.path.exists(results_object_path) or force_redo:
    allopticalResults = aoutils.AllOpticalResults(save_path=results_object_path)
    # make a metainfo attribute to store all metainfo types of info for all experiments/trials
    allopticalResults.metainfo = allopticalResults.slmtargets_stim_responses.loc[:, ['prep_trial', 'date', 'exptype']]


