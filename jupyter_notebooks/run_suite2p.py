# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')

import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as aoutils
import alloptical_plotting as aoplot

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw

date = '2021-01-10'


import sys
sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import os
import numpy as np
from suite2p.run_s2p import run_s2p


# define base path for data and saving results
base_path_data = '/home/pshah/mnt/qnap/Data/2021-01-10'
date = base_path_data[-10:]
base_path_save = '/home/pshah/mnt/qnap/Analysis/%s/suite2p/' % date

import pickle
import numpy as np

date = '2021-01-10'
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-10'

to_suite2p = ['t-002', 't-003', 't-005', 't-007', 't-008', 't-009', 't-010', 't-011', 't-012', 
              't-013', 't-014', 't-015', 't-016']  # specify all trials that were used in the suite2p run

total_frames_stitched = 0
curr_trial_frames = None
baseline_frames = [0, 0]
bad_frames = []
to_suite2p_tiffs = []
for t in to_suite2p:
    pkl_path_2 = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, t, date, t)
    with open(pkl_path_2, 'rb') as f:
        _expobj = pickle.load(f)
        # import suite2p data
    if hasattr(_expobj, 'bad_frames'):
        bad_frames.extend([(int(frame) + total_frames_stitched) for frame in _expobj.bad_frames])
        print(bad_frames[-5:])
    total_frames_stitched += _expobj.n_frames

    to_suite2p_tiffs.append('%s/%s_%s/%s_%s_Cycle00001_Ch3.tif' % (base_path_data, date, t, date, t))

print('# of bad_frames saved to bad_frames.npy: ', len(bad_frames))
np.save(data_path_base + '/bad_frames.npy', np.array(bad_frames))  # save to npy file and remember to move npy file to tiff folder before running with suite2p
# all spont imaging, photostim exp. trials pre and post 4ap


# data_path = []
# for i in to_suite2p:
#     data_path.append('/home/pshah/mnt/qnap/Data/2020-12-19/2020-12-19_%s/' % i)


    
# name of the folder to save results in (default = suite2p in data_path)
save_folder = os.path.join(base_path_save, 'alloptical-2p-08x-alltrials-reg_tiff')  
if not os.path.exists(base_path_save[:-1]):
    os.mkdir(base_path_save)
if not os.path.exists(save_folder):
    print('making the save folder at %s' % save_folder)
    os.mkdir(save_folder)


    
data_path = [os.path.expanduser(base_path_data)]
tiff_list = to_suite2p_tiffs

# setup settings and run suite2p
diameter = 8  # the average diameter (in pixels) of a cell -- check in fiji
fs = 15.  # sampling rate of imaging (default 30 fps)
nplanes = 1  # number of planes (default 1)
nchannels = 1 # number of channels aquired (default 1)  


ops = {
        'batch_size': 2000, # reduce if running out of RAM
        'fast_disk': os.path.expanduser('/mnt/sandbox/pshah/suite2p_tmp'), # used to store temporary binary file, defaults to save_path0 (set as a string NOT a ls)
         #'save_path0': '/media/jamesrowland/DATA/plab/suite_2p', # stores results, defaults to first item in data_path
        'delete_bin': True, # whether to delete binary file after processing
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter': diameter, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1.26, # this is the main parameter for deconvolution (1.25-1.5 for gcamp6s)
        'fs': fs,  # sampling rate (total across planes)
#         'frames_include': 6000,  # just for testing purposes
        # output settings
        'save_mat': True, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 50, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': 0, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True, # whether to register data
        'nimg_init': 200, # subsampled frames for finding reference image
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': True, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        # cell detection settings
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
      }

# # make the local suite2p binaries file if it does not already exist
# if not os.path.exists(ops['fast_disk']):
#     os.mkdir(ops['fast_disk'])
    
# # name of the folder to save results in (default = suite2p in data_path)
# save_folder = os.path.join('/home/pshah/mnt/qnap/Data/2020-12-18/', 'suite2p/spont-2p-LFP-08x')  
# if not os.path.exists(save_folder):
#     os.mkdir(save_folder)



db = {
     'data_path': data_path,
     'tiff_list': tiff_list, 
     'diameter': diameter, 
     'fs': fs,
     'nplanes': nplanes,
     'nchannels': nchannels,
     'save_folder': save_folder
     }


import time as time
# run suite2p
t1 = time.time()
opsEnd=run_s2p(ops=ops,db=db)
t2 = time.time()
print('Total time this cell was running is {}'.format(t2-t1))