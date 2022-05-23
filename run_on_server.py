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

from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
    PhotostimResponsesNonTargetsResults
from _analysis_.nontargets_analysis._ClassPhotostimResponsesAnalysisNonTargets import \
    PhotostimResponsesAnalysisNonTargets

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')

main = PhotostimResponsesAnalysisNonTargets

results: PhotostimResponsesNonTargetsResults = PhotostimResponsesNonTargetsResults.load()

results.collect_nontargets_stim_responses()

results.binned_distances_vs_responses(measurement='new influence response')
# results.binned_distances_vs_responses(measurement='influence response')
# results.binned_distances_vs_responses(measurement='photostim response')





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

