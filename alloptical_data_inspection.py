### various bits of code that is useful for data inspection

import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import draw

# # import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %% IMPORT expobj
expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre h.0')
expobj, experiment = aoutils.import_expobj(prep='RL109', trial='t-020')


# %% useful general plots

aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='Time', figsize=[20, 3])
aoplot.plotLfpSignal(expobj, stim_span_color='', x_axis='time', figsize=[10, 4])
aoplot.plot_SLMtargets_Locs(expobj, background=expobj.meanFluImg_registered)
aoplot.plot_lfp_stims(expobj, x_axis='Time')


# %% s2p ROI Flu trace statistics for each cell

print('s2p neu. corrected cell traces statistics: ')
for cell in expobj.s2p_nontargets:
    cell_idx = expobj.cell_id.index(cell)
    print('mean: %s   \t min: %s  \t max: %s  \t std: %s' %
          (np.round(np.mean(expobj.raw[cell_idx]), 2), np.round(np.min(expobj.raw[cell_idx]), 2), np.round(np.max(expobj.raw[cell_idx]), 2),
           np.round(np.std(expobj.raw[cell_idx], ddof=1), 2)))


