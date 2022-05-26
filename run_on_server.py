#### FILE FOR PUTTING TOGEHTER CODE TO RUN ON THE SERVER
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

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys

from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets
from _analysis_.nontargets_analysis._ClassResultsNontargetPhotostim import PhotostimResponsesNonTargetsResults

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

# %% 3) collecting all summed nontargets photostim and fakestim responses vs. total targets photostim and fakestim responses
# main.run__summed_responses(rerun=1)

# %% 5) responses vs distances

# BASELINE
# results.collect_nontargets_stim_responses(run_pre4ap=True, run_post4ap=False)
# results.binned_distances_vs_responses_baseline(measurement='new influence response')
# results.binned_distances_vs_responses_baseline(measurement='influence response')
# results.binned_distances_vs_responses_baseline(measurement='photostim response')



# INTERICTAL
# results.collect_nontargets_stim_responses(run_pre4ap=False, run_post4ap=True)
# results.binned_distances_vs_responses_interictal(measurement='new influence response')
# results.binned_distances_vs_responses_interictal(measurement='influence response')
# results.binned_distances_vs_responses_interictal(measurement='photostim response')



# ICTAL
results.collect_nontargets_stim_responses(run_pre4ap=False, run_post4ap=False, run_post4ap_ictal=True)
results.binned_distances_vs_responses_ictal(measurement='photostim response')
results.binned_distances_vs_responses_ictal(measurement='new influence response')
results.binned_distances_vs_responses_ictal(measurement='influence response')



# %%








"""# ########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
########### END OF // ZONE FOR CALLING THIS SCRIPT DIRECTLY FROM THE SSH SERVER ###########
"""

sys.exit(0)

