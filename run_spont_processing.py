# turn this all into one .py file that you can run across all photostim. trials

import sys; sys.path.append('/home/pshah/Documents/code/Vape/utils/')
import alloptical_utils_pj as ao
import numpy as np
# import utils_funcs as uf
# import funcs_pj as pjf
import pickle
from paq_utils import frames_discard, paq_read


#%% functions for processing SPONT. IMAGING experiments, used only for making a bad_frames output to use during suite2p

def prep4suite2p(expobj, trial, paths):

    tiff_path_dir = paths[0] % trial
    paq_path = paths[2] % (trial[2:])
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

    tiff_path_dir = paths[0] % trial
    paq_path = paths[2] % (trial[2:])

    print('\n Processing spont. trial # %s' % trial)

    expobj = ao.twopimaging(tiff_path_dir, paq_path)

    prep4suite2p(expobj, trial, paths)

    # Pickle the expobject output to save it for analysis
    pkl_path = paths[4] % (tiff_path_dir, trial)
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("Pkl saved to %s" % pkl_path)

#%% trying out on single trials
# trial = 't-007'
# bad_frames = [[], [], []]
# run_spont_processing(trial, seizure_frames_=[], pkl_path=pkl_path, new_tiffs=new_tiffs, tiffs_loc=tiffs_loc, tiffs_loc2=tiffs_loc2, naparms_loc=naparms_loc, paqs_loc=paqs_loc)


#%% make sure to run EphysViewer.m from MATLAB if you need to specify any bad frames!
trials = ['t-008']

tiffs_loc_dir = '/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_%s'
tiffs_loc = '/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_%s/2020-12-18_%s_Cycle00001_Ch3.tif'
paq_loc = '/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_RL108_%s.paq'
# matlab_loc = '/home/pshah/mnt/qnap/Data/2020-12-18/paired_measurements/2020-12-18_RL108_%s.mat'
matlab_loc = None
pkl_path = '%s/2020-12-18_%s.pkl'

paths = [tiffs_loc_dir, tiffs_loc, paq_loc, matlab_loc, pkl_path]

for trial in trials:
    # prep4suite2p(trial, paths)
    run_spont_processing(trial, paths)

