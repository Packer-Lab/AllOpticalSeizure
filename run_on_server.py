#### FILE FOR PUTTING TOGEHTER CODE TO RUN ON THE SERVER

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')

import alloptical_utils_pj as aoutils

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


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

# prep_trial = 'PS11 t-011'
#
# expobj, _ = aoutils.import_expobj(exp_prep=prep_trial)
#
# print(expobj.responses_SLMtargets_tracedFF_insz)
# print(expobj.responses_SLMtargets_tracedFF_outsz)


### need to re run alloptical processing photostim

# @aoutils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False)
# def run_on_server(**kwargs):
#     expobj = kwargs['expobj']
#     print(expobj.metainfo)
#
#
# run_on_server()



# ### need to re run alloptical processing photostim
#
# @aoutils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True)
# def run_on_server(**kwargs):
#     expobj = kwargs['expobj']
#     # temp run once fully
#     aoutils.run_alloptical_processing_photostim(expobj, plots=False,
#                                                 force_redo=False)
#
# run_on_server()


# aoutils.run_alloptical_processing_photostim(expobj, plots=False)

# %%



sys.exit(0)





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
