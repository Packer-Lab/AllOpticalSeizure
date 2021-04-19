# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects
import os
import numpy as np
import pickle
import tifffile as tf
import sys

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as aoutils


# from Vape.utils.paq_utils import frames_discard, paq_read

# %% update the trial and photostim experiment files information below before running run_photostim_processing()
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-19'
animal_prep = 'PS07'
date = data_path_base[-10:]
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 5 things for every trial
trial = 't-007'  # note that %s magic command in the code below will be using these trials listed here
naparms_loc = '/photostim/2021-01-19_PS07_photostim_013/'  # make sure to include '/' at the end to indicate the child directory
comments = '18 cells x 3 groups; 7mW per cell preset: 2021-01-07_PS_250ms-stim-40hz-multi_interleaved.mat (prot. #3)'
exp_type = 'pre 4ap 2p all optical'  # use 'post' and '4ap' in the description to create the appropriate post4ap exp object
paqs_loc = '%s/%s_PS07_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials
# paqs_loc = '%s/%s_RL111_%s.paq' % (data_path_base, date, '008')  # path to the .paq files for the selected trials
######


tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-17], date, animal_prep, trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
# matlab_badframes_path = None

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

expobj = aoutils.run_photostim_processing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs, metainfo=metainfo,
                         tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc, naparms_loc=(data_path_base+naparms_loc),
                         paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                         processed_tiffs=False, discard_all=True, analysis_save_path=analysis_save_path)


# %% MAKING A BIG bad_frames.npy FILE FOR ALL TRIALS STITCHED TOGETHER (RUN THIS BEFORE RUNNING SUITE2P FOR ALL OPTICAL EXPERIMENTS)

## the code below is run as part of the jupyter notebooks for each experiment's suite2p run
cont = False
if cont:
    # define base path for data and saving results

    import pickle
    import numpy as np

    date = '2021-01-10'
    data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-10'
    base_path_save = '/home/pshah/mnt/qnap/Analysis/%s/suite2p/' % date

    to_suite2p = ['t-002', 't-003', 't-005', 't-007', 't-008', 't-009', 't-010', 't-011', 't-012',
                  't-013', 't-014', 't-015', 't-016']  # specify all trials that were used in the suite2p run
    # to_suite2p = ['t-011', 't-012', 't-013']
    # note ^^^ this only works currently when the spont baseline trials all come first, and also back to back
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

        to_suite2p_tiffs.append('%s/%s_%s/%s_%s_Cycle00001_Ch3.tif' % (data_path_base, date, t, date, t))

    print('# of bad_frames saved to bad_frames.npy: ', len(bad_frames))
    np.save(data_path_base + '/bad_frames.npy', np.array(bad_frames))