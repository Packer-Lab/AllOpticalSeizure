import sys

import numpy as np
from funcsforprajay.funcs import calc_distance_2points

from _analysis_.nontargets_analysis._ClassNonTargetsSzInvasionSpatial import NonTargetsSzInvasionSpatial
from _analysis_.nontargets_analysis._ClassPhotostimResponseQuantificationNonTargets import \
    PhotostimResponsesQuantificationNonTargets
from _analysis_._utils import Results, Quantification
from _utils_._anndata import AnnotatedData2

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import os
from typing import Union

from _utils_ import _alloptical_utils as Utils
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
    """Analysis of response magnitudes of nontargets in relation to spatial distances from seizure boundary"""

    def __init__(self, expobj: Union[alloptical, Post4ap] = None):
        super(NonTargetsResponsesSpatialAnalysis, self).__init__(expobj=expobj)

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=1,
                                    run_post4ap_trials=1,
                                    allow_rerun=0,
                                    skip_trials=PhotostimResponsesQuantificationNonTargets.EXCLUDE_TRIALS,)
    # run_trials=PhotostimResponsesQuantificationNonTargets.TEST_TRIALS)
    def run__initPhotostimResponsesAnalysisNonTargets(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        expobj.NonTargetsResponsesSpatial = NonTargetsResponsesSpatialAnalysis(expobj=expobj)
        expobj.save()

    @staticmethod
    @Utils.run_for_loop_across_exps(run_pre4ap_trials=True, run_post4ap_trials=True, allow_rerun=1,
                                    skip_trials=NonTargetsSzInvasionSpatial.EXCLUDE_TRIALS)
    def run__methods(**kwargs):
        expobj: Union[alloptical, Post4ap] = kwargs['expobj']
        distances_um = expobj.NonTargetsResponsesSpatial._calculate_distance_to_target(expobj=expobj)
        expobj.NonTargetsResponsesSpatial._add_nontargets_distance_to_targets_anndata(expobj.PhotostimResponsesNonTargets.adata, distances_um)
        expobj.save()

    ## CALCULATING DISTANCE TO NEAREST TARGET  #########################################################################
    @staticmethod
    def _add_nontargets_distance_to_targets_anndata(adata: AnnotatedData2, distances):
        adata.add_observation(obs_name='distance to nearest target (um)', values=distances)

    @staticmethod
    def _calculate_distance_to_target(expobj: Union[alloptical, Post4ap]):
        target_coords = expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord']

        distances = []
        for cell in expobj.PhotostimResponsesNonTargets.adata.obs['med']:
            _all_distances = []
            cell_x = cell[1]
            cell_y = cell[0]
            for target in list(target_coords):
                distance = calc_distance_2points((cell_x, cell_y), target)
                _all_distances.append(distance)

            distances.append(np.min(_all_distances))

        distances_um = np.asarray([x / (1 / expobj.pix_sz_x) for x in distances])

        return np.round(distances_um, 3)


if __name__ == '__main__':
    main = NonTargetsResponsesSpatialAnalysis
    results: NonTargetsResponsesSpatialResults = NonTargetsResponsesSpatialResults.load()

    main.run__initPhotostimResponsesAnalysisNonTargets()
    main.run__methods()

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
