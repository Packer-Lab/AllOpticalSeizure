import sys

from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
    PhotostimResponsesQuantificationNonTargets
from _analysis_._utils import Results, Quantification

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os
from typing import Union

import _alloptical_utils as Utils
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


# %% ###### NON TARGETS analysis + plottings

class NonTargetsResponsesSpatialResults(Results):
    SAVE_PATH = SAVE_LOC + 'Results__NonTargetsResponsesSpatialResults.pkl'

    def __init__(self):
        super().__init__()


REMAKE = False
if not os.path.exists(NonTargetsResponsesSpatialResults.SAVE_PATH) or REMAKE:
    results = NonTargetsResponsesSpatialResults()
    results.save_results()

class NonTargetsResponsesSpatialAnalysis(Quantification):
    """Analysis of responses of nontargets in relation to spatial distances from seizure boundary.
    Fig 6.

    - big goal is to use cells binned into proximal, middle or distal.

    [ ] quantification of cells' distance to nearest target.

    """

    def __init__(self, expobj: Post4ap = None):
        super(NonTargetsResponsesSpatialAnalysis, self).__init__(expobj=expobj)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=0,
                                    run_post4ap_trials=1,
                                    allow_rerun=1,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,)
                                    # run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
    def run__initPhotostimResponsesAnalysisNonTargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.NonTargetsResponsesSpatial = NonTargetsResponsesSpatialAnalysis(expobj=expobj)
        expobj.save()






if __name__ == '__main__':
    main = NonTargetsResponsesSpatialAnalysis
    results: NonTargetsResponsesSpatialResults = NonTargetsResponsesSpatialResults.load()



### ARCHIVE

    # # 1) CREATE ANNDATA - create from stims with seizure wavefront, photostim responses, distances of nontargets to sz wavefront as layer, add significant responders infor from baseline and ictal stims analysed,
    # def create_anndata(self, expobj: Post4ap):
    #     """
    #     create anndata table to store information about nontargets sz invasion spatial from:
    #
    #     stims with seizure wavefront,
    #     photostim responses,
    #     distances of nontargets to sz wavefront as layer,
    #     add significant responders infor from baseline and ictal stims analysed,
    #
    #
    #     :param expobj:
    #     """
    #     assert hasattr(expobj, 'NonTargetsSzInvasionSpatial'), 'nontargets sz invasion spatial processing not found for exp obj.'



