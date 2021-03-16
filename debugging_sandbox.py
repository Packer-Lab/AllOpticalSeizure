# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')



# %run ./setup_notebook.ipynb
# print(sys.path)
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot
from utils import funcs_pj as pj

import pickle

###### IMPORT pkl file containing expobj
trial = 't-011'
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
with open(pkl_path, 'rb') as f:
    print('importing expobj for "%s" from: %s' % (date, pkl_path))
    expobj = pickle.load(f)
    experiment = '%s: %s, %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])
    print('DONE IMPORT of %s' % experiment)

# %%
expobj.targets_dff, expobj.targets_dff_avg, expobj.targets_dfstdF, \
    expobj.targets_dfstdF_avg, expobj.targets_raw, expobj.targets_raw_avg = \
    aoutils.get_targets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim=expobj.pre_stim,
                                         post_stim=expobj.post_stim)
