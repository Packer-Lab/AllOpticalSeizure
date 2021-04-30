# turn this all into one .py file that you can run across all photostim. trials

import os
import sys; sys.path.append('/home/pshah/Documents/code/Vape/utils/')
import alloptical_utils_pj as aoutils
import numpy as np
import pickle
from utils.paq_utils import frames_discard, paq_read


#%% functions for processing SPONT. IMAGING experiments, used only for making a bad_frames output to use during suite2p

def prep4suite2p(expobj, trial, paths, discard_all):

    tiff_path_dir = paths[0]
    paq_path = paths[2]
    # expobj = ao.TwoPhotonImaging(tiff_path_dir, paq_path)

    if paths[3] is not None:
        paq = paq_read(file_path=paq_path, plot=False)
        print(paq[0]['data'][0])
        bad_frames = frames_discard(paq=paq[0], input_array=paths[3] % (trial[2:]), total_frames=expobj.n_frames)
        print('\n', bad_frames)
        print('Total photostim and/or seizure/CSD frames: ', len(bad_frames))
    elif discard_all:
        # add all frames as bad frames incase want to include this trial in suite2p run
        paq = paq_read(file_path=paq_path, plot=False)
        bad_frames = frames_discard(paq=paq[0], input_array=None, total_frames=expobj.n_frames, discard_all=True)
        print('\n', bad_frames)
        print('Total photostim and/or seizure/CSD frames: ', len(bad_frames))
    else:
        bad_frames = []
        print('No bad frames needed for', tiff_path_dir)

    if len(bad_frames) > 0:
        np.save('%s/bad_frames.npy' % tiff_path_dir,
                bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p
    return bad_frames

def run_spont_processing(trial, paths, analysis_save_path, metainfo, discard_all):

    tiff_path_dir = paths[0]
    tiff_path = paths[1]
    paq_path = paths[2]

    print('\n Processing spont. trial # %s' % trial)

    expobj = aoutils.TwoPhotonImaging(tiff_path_dir, tiff_path, paq_path, metainfo, analysis_save_path=analysis_save_path)

    # add all frames as bad frames incase want to include this trial in suite2p run
    paq = paq_read(file_path=paq_path, plot=False)
    expobj.bad_frames = frames_discard(paq=paq[0], input_array=None, total_frames=expobj.n_frames, discard_all=discard_all)

    expobj.bad_frames = prep4suite2p(expobj, trial, paths, discard_all=discard_all)

    # set analysis save path for expobj
    # make the necessary Analysis saving subfolder as well
    expobj.analysis_save_path = analysis_save_path
    if os.path.exists(expobj.analysis_save_path):
        pass
    elif os.path.exists(expobj.analysis_save_path[:-17]):
        os.mkdir(expobj.analysis_save_path)
    elif os.path.exists(expobj.analysis_save_path[:-27]):
        os.mkdir(expobj.analysis_save_path[:-17])
        os.mkdir(expobj.analysis_save_path)


    # Pickle the expobject output to save it for analysis
    pkl_path = paths[4]
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("Pkl saved to %s" % pkl_path)


#%% make sure to run EphysViewer.m from MATLAB if you need to specify any bad frames!
# trial = 't-001'
trials = ['t-019', 't-020']
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-09'
animal_prep = 'PS04'
date = data_path_base[-10:]
exp_type = 'spont imaging'
comments = 'spont imaging period before running alloptical experiment'


for trial in trials:
    metainfo = {
        'animal prep.': animal_prep,
        'trial': trial,
        'date': date,
        'exptype': exp_type,
        'data_path_base': data_path_base,
        'comments': comments
    }
    paqs_loc = '%s/%s_%s_%s.paq' % (data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials


    tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
    tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
    # matlab_loc = '/home/pshah/mnt/qnap/Data/2020-12-18/paired_measurements/2020-12-18_RL108_%s.mat'
    matlab_loc = None
    discard_all = True
    analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]


    paths = [tiffs_loc_dir, tiffs_loc, paqs_loc, matlab_loc, pkl_path]

    run_spont_processing(trial, paths, analysis_save_path=analysis_save_path, metainfo=metainfo, discard_all=discard_all)