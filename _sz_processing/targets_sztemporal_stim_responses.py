"""TODO/GOALS:
2) plotting average traces around time of seizure invasion for all targets across all exps
    - plot also the mean FOV Flu at the bottom
3) plot average stim response before and after time of seizure invasion for all targets across all exps

"""
import matplotlib.pyplot as plt
import pandas as pd

import _alloptical_utils as Utils

from _main_.Post4apMain import Post4ap

# SAVE_LOC = "/Users/prajayshah/OneDrive/UTPhD/2022/OXFORD/export/"
from _sz_processing.temporal_delay_to_sz_invasion import convert_timedel2frames

SAVE_LOC = "/home/pshah/mnt/qnap/Analysis/analysis_export/"

expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')

# expobj.slmtargets_data.var[['stim_start_frame', 'wvfront in sz', 'seizure_num']]

# %% plotting seizure invasion Flu traces from targets


for target, coord in enumerate(expobj.slmtargets_data.obs['SLM target coord']):
    target, coord = 0, expobj.slmtargets_data.obs['SLM target coord'][0]
    cols_ = [idx for idx, col in enumerate([*expobj.slmtargets_data.obs]) if 'time_del' in col]
    sz_times = expobj.slmtargets_data.obs.iloc[target, cols_]
    fr_times = [convert_timedel2frames(expobj, sznum, time) for sznum, time in enumerate(sz_times) if not pd.isnull(time)]

    # plot each frame seizure invasion time Flu
    fig, ax = plt.subplots(figsize=[2, 4])
    for fr in fr_times:
        to_plot = expobj.raw_SLMTargets[target][fr-int(2*expobj.fps): fr+int(2*expobj.fps)]
        ax.plot(to_plot)
    fig.show()




