from typing import List

import numpy as np
import pandas as pd
from funcsforprajay.wrappers import plot_piping_decorator
from matplotlib import pyplot as plt

import _alloptical_utils as Utils
import funcsforprajay.funcs as pj

from _analysis_._utils import Quantification
from _main_.Post4apMain import Post4ap
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames


class TargetsSzInvasionSpatial(Quantification):
    range_of_sz_spatial_distance: List[float] = None  # TODO need to collect - represents the 25th, 50th, and 75th percentile range of the sz invasion distance stats calculated across all targets and all exps - maybe each seizure across all exps should be the 'n'?

    def __init__(self, expobj: Post4ap):
        super().__init__(expobj)
        print(f'\- ADDING NEW TargetsSzInvasionSpatial MODULE to expobj: {expobj.t_series_name}')

    def __repr__(self):
        return f"TargetsSzInvasionSpatial <-- Quantification Analysis submodule for expobj <{self.expobj_id}>"
