# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects

import numpy as np
import pickle
import tifffile as tf
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as ao
from paq_utils import frames_discard, paq_read


# from Vape.utils.paq_utils import frames_discard, paq_read


# for processing PHOTOSTIM. experiments, creates the all-optical expobj saved in a pkl files at imaging tiff's loc

def rm_artifacts_tiffs(expobj, tiffs_loc, new_tiffs):
    ### make a new tiff file (not for suite2p) with the first photostim frame whitened, and save new tiff
    print('\n-----making processed photostim .tiff from:')
    tiff_path = tiffs_loc
    print(tiff_path)
    im_stack = tf.imread(tiff_path, key=range(expobj.n_frames))
    print('Processing experiment tiff of shape: ', im_stack.shape)

    frames_to_whiten = []
    for j in expobj.stim_start_frames:
        frames_to_whiten.append(j)

    # number of photostim frames with artifacts
    frames_to_remove = []
    for j in expobj.stim_start_frames:
        for i in range(0,
                       expobj.duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
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

    save_path = (new_tiffs + "_artifactrm.tif")
    tf.imwrite(save_path, im_stack_1, photometric='minisblack')

    del im_stack_1

    # draw areas on top of im_stack_1 where targets are:
    im_stack_2 = im_stack
    print('Shape', im_stack_2.shape)

    for stim in range(expobj.n_groups):
        b = np.full_like(im_stack_2[0], fill_value=0)
        targets = expobj.target_areas[stim]
        for i in np.arange(len(targets)):
            for j in targets[i]:
                b[j] = 5000

        all_stim_start_frames = []
        for stim_frame in expobj.stim_start_frames[stim::expobj.n_groups]:
            all_stim_start_frames.append(stim_frame)
        for frame in all_stim_start_frames:
            #         im_stack_2[frame-4] = im_stack_2[frame-4]+b
            #         im_stack_2[frame-3] = im_stack_2[frame-3]+b
            #        im_stack_2[frame-2] = im_stack_2[frame-2]+b
            im_stack_2[frame - 1] = im_stack_2[frame - 1] + b

    im_stack_2 = np.delete(im_stack_2, expobj.photostim_frames, axis=0)

    print('After delete shape targetcells', im_stack_2.shape)

    save_path = (new_tiffs + '_targetcells.tif')
    tf.imwrite(save_path, im_stack_2, photometric='minisblack')

    print('done saving to: ', save_path)

    del im_stack_2
    del im_stack


def run_photostim_processing(trial, exp_type, tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc, pkl_path, seizure_comments,
                             new_tiffs, matlab_badframes_path=None, processed_tiffs=True, discard_all=False):

    print('\n-----Processing trial # %s-----' % trial)

    paths = [[tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc]]
    # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

    if 'post' in exp_type and '4ap' in exp_type:
        expobj = ao.Post4ap(paths[0], stimtype='2pstim')
    elif 'pre' in exp_type and '4ap' in exp_type:
        expobj = ao.alloptical(paths[0], stimtype='2pstim')
    else:
        expobj = ao.twopimaging()

    # for key, values in vars(expobj).items():
    #     print(key)

    # expobj._parseNAPARMxml()
    # expobj._parseNAPARMgpl()
    # expobj._parsePVMetadata()
    expobj.stimProcessing(stim_channel='markpoints2packio')
    expobj._findTargets()
    expobj.find_photostim_frames()

    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("\nPkl saved to %s" % pkl_path)

    # collect information about seizures
    if 'post' in exp_type and '4ap' in exp_type:
        expobj.collect_seizures_info(seizures_info_array=matlab_badframes_path, seizure_comments=comments,
                                     discard_all=discard_all)

    # if matlab_badframes_path is not None or discard_all is True:
    #     paq = paq_read(file_path=paqs_loc, plot=False)
    #     # print(paq[0]['data'][0])  # print the frame clock signal from the .paq file to make sure its being read properly
    #     bad_frames, expobj.seizure_frames, _, _ = \
    #         frames_discard(paq=paq[0], input_array=matlab_badframes_path,
    #                        total_frames=expobj.n_frames, discard_all=discard_all)
    #     print('\nTotal extra seizure/CSD or other frames to discard: ', len(bad_frames))
    #     print('|\n -- first and last 10 indexes of these frames', bad_frames[:10], bad_frames[-10:])
    #     expobj.append_bad_frames(
    #         bad_frames=bad_frames)  # here only need to append the bad frames to the expobj.bad_frames property
    #
    # else:
    #     expobj.seizure_frames = []
    #     print('\nNo additional bad (seizure) frames needed for', tiffs_loc_dir)
    #
    # if len(expobj.bad_frames) > 0:
    #     print('***Saving a total of ', len(expobj.bad_frames),
    #           'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy***')
    #     np.save('%s/bad_frames.npy' % tiffs_loc_dir,
    #             expobj.bad_frames)  # save to npy file and remember to move npy file to tiff folder before running with suite2p

    # Pickle the expobject output to save it for analysis

    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("\nPkl saved to %s" % pkl_path)

    # make processed tiffs
    if processed_tiffs:
        rm_artifacts_tiffs(expobj, tiffs_loc=tiffs_loc, new_tiffs=new_tiffs)

    print('\n----- COMPLETED RUNNING run_photostim_processing() -----')


# %% update the trial and photostim experiment files information below before running run_photostim_processing()
trial = 't-013'  # note that %s magic command in the code below will be using these trials listed here

data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-18'
date = '2020-12-18'

# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
naparms_loc = '%s/photostim/2020-12-18_RL108_ps_008/' % data_path_base  # make sure to include '/' at the end to indicate the child directory

exp_type = 'post 4ap'
comments = '5 seizure events on LFP (trial starts during a seizure)'
tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = '%s/%s_%s.pkl' % (tiffs_loc_dir, date, trial)  # specify path to save pkl object
paqs_loc = '%s/%s_RL108_%s.paq' % (
data_path_base, date, trial[2:])  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved

# choose matlab path if need to use or use None for no additional bad frames
matlab_badframes_path = '%s/paired_measurements/2020-12-18_RL108_%s.mat' % (data_path_base, trial[2:])
# matlab_badframes_path = None

run_photostim_processing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs, seizure_comments=comments,
                         tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc, naparms_loc=naparms_loc,
                         paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                         processed_tiffs=False, discard_all=True)
