#%% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import numpy as np
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj


# import onePstim superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = aoutils.import_resultsobj(pkl_path=results_object_path)


# %%
###### IMPORT pkl file containing data in form of expobj
trial = 't-010'
date = '2021-01-08'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date)


# %% ########## BAR PLOT showing average success rate of photostimulation

# plot across different groups
t009_pre_4ap_reliability = list(expobj.StimSuccessRate_SLMtargets.values())
# t011_post_4ap_reliabilty = list(expobj.StimSuccessRate_cells.values())  # reimport another expobj for post4ap trial
t013_post_4ap_reliabilty = list(expobj.StimSuccessRate_SLMtargets.values())  # reimport another expobj for post4ap trial
#
pj.plot_bar_with_points(data=[t009_pre_4ap_reliability, t013_post_4ap_reliabilty], xlims=[0.25, 0.3],
                        x_tick_labels=['pre-4ap', 'post-4ap'], colors=['green', 'deeppink'], y_label='% success stims.',
                        ylims=[0, 100], bar=False, title='success rate of stim. responses', expand_size_y=1.2, expand_size_x=1.2)

