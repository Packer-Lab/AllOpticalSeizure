#### FILE FOR PUTTING TOGEHTER CODE TO RUN ON THE SERVER

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys; import os
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj


# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/SLMtargets_responses_2021-11-23'
os.makedirs(save_path_prefix) if not os.path.exists(save_path_prefix) else None


# expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-013')

"""######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
######### ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""

# %% troubleshoot

prep_trial = 'PS11 t-011'

expobj, _ = aoutils.import_expobj(exp_prep=prep_trial)

print(expobj.responses_SLMtargets_tracedFF_insz)
print(expobj.responses_SLMtargets_tracedFF_outsz)


### need to re run alloptical processing photostim

# @aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False)
# def run_on_server(**kwargs):
#     expobj = kwargs['expobj']
#     print(expobj.metainfo)
#
#
# run_on_server()



### need to re run alloptical processing photostim

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def run_on_server(**kwargs):
    expobj = kwargs['expobj']
    # temp run once fully
    aoutils.run_alloptical_processing_photostim(expobj, plots=False,
                                                force_redo=False)

run_on_server()


# aoutils.run_alloptical_processing_photostim(expobj, plots=False)

# %% aoanalysis-photostim-SLMtargets-1) adding slm targets responses to alloptical results allopticalResults.slmtargets_stim_responses

@aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
def add_slmtargets_responses_tracedFF(**kwargs):
    print("|- adding slm targets trace dFF responses to allopticalResults.slmtargets_stim_responses")
    print(kwargs)
    expobj = kwargs['expobj'] if 'expobj' in kwargs.keys() else KeyError('need to provide expobj as keyword argument')

    if 'pre' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets)'] = dFprestimF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF.loc[i, :]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

        print(f"|- {prep_trial}: delta trace dFF response: {delta_trace_dFF_response:.2f}, reliability: {reliability:.2f},  dFprestimF_response: {dFprestimF_response:.2f}")

    elif 'post' in expobj.metainfo['exptype']:
        prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"
        dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf_outsz[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets)'] = dFstdF_response

        dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf_outsz.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets)'] = dFprestimF_response

        reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_tracedFF_outsz.values()))
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF))'] = reliability

        delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF_outsz.loc[i, :]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets)'] = delta_trace_dFF_response

        print(f"|- {prep_trial} (outsz): delta trace dFF response: {delta_trace_dFF_response:.2f}, reliability: {reliability:.2f},  dFprestimF_response: {dFprestimF_response:.2f}")

        if len(expobj.stims_in_sz) > 0:
            prep_trial = f"{expobj.metainfo['animal prep.']} {expobj.metainfo['trial']}"

            dFstdF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfstdf_insz[i]) for i in range(expobj.n_targets_total)]])  # these are not dFstdF responses right now!!!
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/stdF all targets) insz'] = dFstdF_response

            dFprestimF_response = np.mean([[np.mean(expobj.responses_SLMtargets_dfprestimf_insz.loc[i, :]) for i in range(expobj.n_targets_total)]])  #
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (dF/prestimF all targets) insz'] = dFprestimF_response

            reliability = np.mean(list(expobj.StimSuccessRate_SLMtargets_insz.values()))
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean reliability (>10 delta(trace_dFF)) insz'] = reliability

            delta_trace_dFF_response = np.mean([[np.mean(expobj.responses_SLMtargets_tracedFF_insz.loc[i, :]) for i in range(expobj.n_targets_total)]])
            allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses['prep_trial'] == prep_trial, 'mean response (delta(trace_dFF) all targets) insz'] = delta_trace_dFF_response

            print(f"|- {prep_trial}: delta trace dFF response (in sz): {delta_trace_dFF_response:.2f}, reliability (in sz): {reliability:.2f},  dFprestimF_response (in sz): {dFprestimF_response:.2f}")

add_slmtargets_responses_tracedFF()

allopticalResults.slmtargets_stim_responses
allopticalResults.save()

## check allopticalResults.slmtargets_stim_responses
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.post_4ap_trials))]['mean response (delta(trace_dFF) all targets)']
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.post_4ap_trials))]['mean response (delta(trace_dFF) all targets) insz']
allopticalResults.slmtargets_stim_responses[allopticalResults.slmtargets_stim_responses['prep_trial'].isin(pj.flattenOnce(allopticalResults.pre_4ap_trials))]['mean response (delta(trace_dFF) all targets)']






"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""

sys.exit()
