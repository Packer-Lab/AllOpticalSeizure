# turn this all into one .py file that you can run across all photostim. trials

import os
import sys; sys.path.append('/home/pshah/Documents/code/Vape/utils/')
import alloptical_utils_pj as ao
import numpy as np
# import utils_funcs as uf
# import funcs_pj as pjf
import pickle
from utils.paq_utils import frames_discard, paq_read


#%% functions for processing SPONT. IMAGING experiments, used only for making a bad_frames output to use during suite2p

def prep4suite2p(expobj, trial, paths):

    tiff_path_dir = paths[0]
    paq_path = paths[2]
    # expobj = ao.twopimaging(tiff_path_dir, paq_path)

    if paths[3] is not None:
        paq = paq_read(file_path=paq_path, plot=False)
        print(paq[0]['data'][0])
        bad_frames = frames_discard(paq=paq[0], input_array=paths[3] % (trial[2:]), total_frames=expobj.n_frames)
        print('\n', bad_frames)
        print('Total photostim and/or seizure/CSD frames: ', len(bad_frames))

    else:
        bad_frames = []
        print('No bad frames needed for', tiff_path_dir)

    if len(bad_frames) > 0:
        np.save('%s/bad_frames.npy' % tiff_path_dir,
                bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p

def run_spont_processing(trial, paths):

    tiff_path_dir = paths[0]
    paq_path = paths[2]

    print('\n Processing spont. trial # %s' % trial)

    expobj = ao.twopimaging(tiff_path_dir, paq_path)

    prep4suite2p(expobj, trial, paths)

    # set analysis save path for expobj
    # make the necessary Analysis saving subfolder as well
    expobj.analysis_save_path = analysis_save_path
    if os.path.exists(expobj.analysis_save_path):
        pass
    elif os.path.exists(expobj.analysis_save_path[:-17]):
        os.mkdir(expobj.analysis_save_path)
    elif os.path.exists(expobj.analysis_save_path[:-27]):
        os.mkdir(expobj.analysis_save_path[:-17])


    # Pickle the expobject output to save it for analysis
    pkl_path = paths[4]
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("Pkl saved to %s" % pkl_path)


#%% make sure to run EphysViewer.m from MATLAB if you need to specify any bad frames!
trial = 't-006'
data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
animal_prep = 'RL109'
date = '2020-12-19'
paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials


tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# matlab_loc = '/home/pshah/mnt/qnap/Data/2020-12-18/paired_measurements/2020-12-18_RL108_%s.mat'
matlab_loc = None
analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]


paths = [tiffs_loc_dir, tiffs_loc, paqs_loc, matlab_loc, pkl_path]

run_spont_processing(trial, paths)

