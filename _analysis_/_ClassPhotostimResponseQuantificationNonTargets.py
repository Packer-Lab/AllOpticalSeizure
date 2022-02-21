from typing import Union, List

import numpy as np
import pandas as pd

import _alloptical_utils as Utils
from _analysis_._utils import Quantification
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from funcsforprajay import plotting as pplot

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _utils_._anndata import AnnotatedData2

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/analysis_quantification_classes/"


class PhotostimResponsesQuantificationNonTargets(Quantification):
    save_path = SAVE_LOC + 'PhotostimResponsesQuantificationNonTargets.pkl'
    mean_photostim_responses_baseline: List[float] = None
    mean_photostim_responses_interictal: List[float] = None
    mean_photostim_responses_ictal: List[float] = None

    def __init__(self, expobj: alloptical):
        super().__init__(expobj)
        print(f'\- ADDING NEW PhotostimResponsesNonTargets MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"PhotostimResponsesNonTargets <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"
