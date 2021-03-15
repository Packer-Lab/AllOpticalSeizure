# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')



# %run ./setup_notebook.ipynb
# print(sys.path)
from utils import funcs_pj as pj

import pickle

###### IMPORT pkl file containing expobj
trial = 't-011'
experiment = 'RL108: photostim-post4ap-%s' % trial
date = '2020-12-18'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

with open(pkl_path, 'rb') as f:
    expobj = pickle.load(f)
print('imported expobj for "%s %s" from: %s' % (date, experiment, pkl_path))

# %%

stims_of_interest = [1424, 1572, 1720, 1868]
flip_stims = [1424, 1572, 1720]

expobj.cells_sz_stim = {}
for stim in stims_of_interest:
    sz_border_path = "%s/boundary_csv/2020-12-18_%s_stim-%s.tif_border.csv" % (expobj.analysis_save_path, trial, stim)
    if stim in flip_stims:
        flip = True
    else:
        flip = False
    in_sz = expobj.classify_cells_sz(sz_border_path, to_plot=True, title='%s' % stim, flip=flip)
    expobj.cells_sz_stim[stim] = in_sz  # for each stim, there will be a list of cells that will be classified as in seizure or out of seizure

