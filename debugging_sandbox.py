import numpy as np
import matplotlib.pyplot as plt

# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')



# %run ./setup_notebook.ipynb
# print(sys.path)
import funcs_pj as pj


import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
import pickle

###### IMPORT pkl file containing expobj
trial = 't-009'
experiment = 'RL108: photostim-pre4ap-%s' % trial
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

with open(pkl_path, 'rb') as f:
    expobj = pickle.load(f)
print('imported expobj for "%s %s" from: %s' % (date, experiment, pkl_path))

# %%


group1 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'photostim target']['Avg. dF/stdF response'])
group2 = list(expobj.average_responses_dfstdf[expobj.average_responses_dfstdf['group'] == 'non-target']['Avg. dF/stdF response'])
pj.bar_with_points(data=[group1, group2], x_tick_labels=['photostim target', 'non-target'], xlims=[0, 0.6], ylims=[0, 1.5], bar=False,
                   colors=['red', 'black'], title=experiment, y_label='Avg dF/stdF response', expand_size_y=1.3, expand_size_x=1.5)
