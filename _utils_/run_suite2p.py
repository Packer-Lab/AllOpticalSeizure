import os; os.getcwd()
from suite2p.run_s2p import run_s2p
import numpy as np
#%%
# tiffs to run s2p on
to_suite2p = ['t-009', 't-010', 't-011']
save_folder = os.path.join('/home/pshah/mnt/qnap/Data/2020-03-17/RL077/', 'suite2p/photostim-baseline')  # name of the folder to save results in (default = suite2p in data_path)


to_suite2p_tiffs = []
for i in to_suite2p:
    to_suite2p_tiffs.append('/home/pshah/mnt/qnap/Data/2020-03-17/RL077/2020-03-17_%s/2020-03-17_%s_Cycle00001_Ch3.tif' % (i,i))

data_path = []
for i in to_suite2p:
    data_path.append('/home/pshah/mnt/qnap/Data/2020-03-17/RL077/2020-03-17_%s/' % i)

tiff_list = to_suite2p_tiffs

#%%
# setup settings for suite2p
cell_diameter = 5.5
imaging_fps = 30.

ops = {
    'batch_size': 2000,  # reduce if running out of RAM
    'fast_disk': os.path.expanduser('~/Documents/suite2p_binaries'),
    # used to store temporary binary file, defaults to save_path0 (set as a string NOT a ls)
    # 'save_path0': '/media/jamesrowland/DATA/plab/suite_2p', # stores results, defaults to first item in data_path
    'delete_bin': True,  # whether to delete binary file after processing
    # main settings
    'nplanes': 1,  # each tiff has these many planes in sequence
    'nchannels': 1,  # each tiff has these many channels per plane
    'functional_chan': 1,  # this channel is used to extract functional ROIs (1-based)
    'diameter': cell_diameter,
    # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
    'tau': 1.26,  # this is the main parameter for deconvolution (1.25-1.5 for gcamp6s)
    'fs': imaging_fps,  # sampling rate (total across planes)
    # output settings
    'save_mat': True,  # whether to save output as matlab files
    'combined': True,  # combine multiple planes into a single result /single canvas for GUI
    # parallel settings
    'num_workers': 50,  # 0 to select num_cores, -1 to disable parallelism, N to enforce value
    'num_workers_roi': 0,  # 0 to select number of planes, -1 to disable parallelism, N to enforce value
    # registration settings
    'do_registration': True,  # whether to register data
    'nimg_init': 200,  # subsampled frames for finding reference image
    'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
    'align_by_chan': 1,  # when multi-channel, you can align by non-functional channel (1-based)
    'reg_tif': False,  # whether to save registered tiffs
    'subpixel': 10,  # precision of subpixel registration (1/subpixel steps)
    # 'two_step_registration': True,
    # 'keep_movie_raw': True,
    # cell detection settings
    'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
    'navg_frames_svd': 5000,  # max number of binned frames for the SVD
    'nsvd_for_roi': 1000,  # max number of SVD components to keep for ROI detection
    'max_iterations': 20,  # maximum number of iterations to do cell detection
    'ratio_neuropil': 6.,  # ratio between neuropil basis size and cell radius
    'ratio_neuropil_to_cell': 3,  # minimum ratio between neuropil radius and cell radius
    'tile_factor': 1.,  # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
    'threshold_scaling': 1.,  # adjust the automatically determined threshold by this scalar multiplier
    'max_overlap': 0.75,  # cells with more overlap than this get removed during triage, before refinement
    'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
    'outer_neuropil_radius': np.inf,  # maximum neuropil radius
    'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
    # deconvolution settings
    'baseline': 'maximin',  # baselining mode
    'win_baseline': 60.,  # window for maximin
    'sig_baseline': 10.,  # smoothing constant for gaussian filter
    'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
    'neucoeff': .7,  # neuropil coefficient
}

# make the local_data_path suite2p binaries file if it does not already exist
if not os.path.exists(ops['fast_disk']):
    os.mkdir(ops['fast_disk'])

diameter = cell_diameter  # the average diameter (in pixels) of a cell -- check in fiji
fs = int(imaging_fps)  # sampling rate of imaging (default 30 fps)
nplanes = 1  # number of planes (default 1)
nchannels = 1  # number of channels aquired (default 1)

db = {
    'data_path': data_path,
    'tiff_list': tiff_list,
    'diameter': diameter,
    'fs': fs,
    'nplanes': nplanes,
    'nchannels': nchannels,
    'save_folder': save_folder
}

#%%
import time as time
# run suite2p
t1 = time.time()
opsEnd=run_s2p(ops=ops, db=db)
t2 = time.time()
print('Total time this cell was running is {}'.format(t2-t1))


