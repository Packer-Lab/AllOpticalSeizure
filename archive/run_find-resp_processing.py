# turn this all into one .py file that you can run across all photostim. trials

import sys; sys.path.append('/home/pshah/Documents/code/Vape/_utils_/')
import alloptical_utils_pj as ao
import numpy as np
import pickle

#%% functions for processing FIND RESPONDERS experiments, creates a pkl files that is saved in the imaging tiff's loc

def rm_artifacts_tiffs(exp_obj, tiffs_loc, tiffs_loc2, new_tiffs):
    ### make a new tiff file (not for suite2p) with the first photostim frame whitened, and save new tiff
    import tifffile as tf

    tiff_path = tiffs_loc2
    print(tiff_path)
    im_stack = tf.imread(tiff_path, key=range(exp_obj.n_frames))
    print('\nProcessing naparm tiff of shape: ', im_stack.shape)

    frames_to_whiten = []
    for j in exp_obj.stim_start_frames[0]:
        frames_to_whiten.append(j)

    # numba of photostim frames with artifacts
    frames_to_remove = []
    for j in exp_obj.stim_start_frames[0]:
        for i in range(0,
                       exp_obj.stim_duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
            frames_to_remove.append(j + i)

    print('# of total photostim artifact frames:', len(frames_to_remove))

    im_stack_1 = im_stack
    a = np.full_like(im_stack_1[0], fill_value=0)
    a[0:100, 0:100] = 5000.
    for frame in frames_to_whiten:
        im_stack_1[frame - 3] = im_stack_1[frame - 3] + a
        im_stack_1[frame - 2] = im_stack_1[frame - 2] + a
        im_stack_1[frame - 1] = im_stack_1[frame - 1] + a

    print('Shape', im_stack_1.shape)
    im_stack_1 = np.delete(im_stack_1, frames_to_remove, axis=0)
    print('After delete shape artifactrem', im_stack_1.shape)

    save_path = (new_tiffs % (trial, trial) + "_artifactrm.tif")
    tf.imwrite(save_path, im_stack_1, photometric='minisblack')

    del im_stack_1

    # draw areas on top of im_stack_1 where targets are:
    im_stack_2 = im_stack
    print('Shape', im_stack_2.shape)

    for stim in range(exp_obj.n_groups):
        b = np.full_like(im_stack_2[0], fill_value=0)
        targets = exp_obj.target_areas[stim]; print(len(targets))
        for i in np.arange(len(targets)):
            for j in targets[i]:
                print(targets[i])
                b[j] = 5000

        all_stim_start_frames = []
        for stim_frame in exp_obj.stim_start_frames[0][stim::exp_obj.n_groups]:
            all_stim_start_frames.append(stim_frame)
        for frame in all_stim_start_frames:
            #         im_stack_2[frame-4] = im_stack_2[frame-4]+b
            #         im_stack_2[frame-3] = im_stack_2[frame-3]+b
            #        im_stack_2[frame-2] = im_stack_2[frame-2]+b
            im_stack_2[frame - 1] = im_stack_2[frame - 1] + b

    im_stack_2 = np.delete(im_stack_2, exp_obj.photostim_frames, axis=0)

    print('After delete shape targetcells', im_stack_2.shape)

    save_path = (new_tiffs % (trial, trial) + "_targetcells.tif")
    tf.imwrite(save_path, im_stack_2, photometric='minisblack')

    print('done saving to: ', save_path)

    del im_stack_2
    del im_stack

def run_find_resp_processing(trial, tiffs_loc, tiffs_loc2, naparms_loc, paqs_loc, seizure_frames_, pkl_path, new_tiffs):
    tiffs_loc = tiffs_loc % trial
    tiffs_loc2 = tiffs_loc2 % (trial, trial)
    naparms_loc = naparms_loc
    paqs_loc = paqs_loc % trial

    # tiffs_loc = '/home/pshah/mnt/qnap/Data/2020-03-18/J063/2020-03-18_J063_%s' % trial
    # tiffs_loc2 = '/home/pshah/mnt/qnap/Data/2020-03-18/J063/2020-03-18_J063_%s/2020-03-18_J063_%s_Cycle00001_Ch3.tif' % (trial, trial)
    # naparms_loc = '/home/pshah/mnt/qnap/Data/2020-03-18/J063/photostim/2020-03-18_photostim_002'
    # paqs_loc = '/home/pshah/mnt/qnap/Data/2020-03-18/2020-03-18_J063_%s.paq' % trial
    print('\n-----Processing trial # %s-----' % trial)

    paths = [[tiffs_loc, naparms_loc, paqs_loc]]
    print(paths)

    exp_obj = ao.alloptical(paths[0], stim='2pstim')
    for key, values in vars(exp_obj).items():
        print(key)

    exp_obj._parseNAPARMgpl()
    exp_obj._parseNAPARMxml()
    exp_obj._parsePVMetadata()
    exp_obj.stimProcessing(stim_channel='markpoints2packio')

    exp_obj._findTargets_naparm()
    exp_obj.find_photostim_frames()
    exp_obj.append_seizure_frames(bad_frames=seizure_frames_)

    print('Total photostim + seizure/CSD frames: ', len(exp_obj.bad_frames))
    np.save('%s/bad_frames.npy' % tiffs_loc,
            exp_obj.bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p

    # Pickle the expobject output to save it for analysis
    pkl_path = pkl_path % (tiffs_loc, trial)
    with open(pkl_path, 'wb') as f:
        pickle.dump(exp_obj, f)
    print("Pkl saved to %s" % pkl_path)

    # make processed tiffs
    rm_artifacts_tiffs(exp_obj, tiffs_loc=tiffs_loc, tiffs_loc2=tiffs_loc2, new_tiffs=new_tiffs)

#%% trying out on single trials
naparms_loc = '/home/pshah/mnt/qnap/Data/2020-03-17/RL076/photostim/2020-03-17_find_resp_003'

pkl_path = '%s/2020-03-17_%s.pkl'

tiffs_loc = '/home/pshah/mnt/qnap/Data/2020-03-17/RL076/2020-03-17_%s'
tiffs_loc2 = '/home/pshah/mnt/qnap/Data/2020-03-17/RL076/2020-03-17_%s/2020-03-17_%s_Cycle00001_Ch3.tif'
paqs_loc = '/home/pshah/mnt/qnap/Data/2020-03-17/RL076/2020-03-17_%s.paq'
new_tiffs = '/home/pshah/mnt/qnap/Data/2020-03-17/RL076/2020-03-17_%s/2020-03-17_%s'


trial = 't-014'
bad_frames = []
run_find_resp_processing(trial, seizure_frames_=bad_frames, pkl_path=pkl_path, new_tiffs=new_tiffs,
                             tiffs_loc=tiffs_loc, tiffs_loc2=tiffs_loc2, naparms_loc=naparms_loc, paqs_loc=paqs_loc)

