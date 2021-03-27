# Step #1) in all optical experiment analysis - preprocessing the data to prep for suite2p analysis and creating some starter experiment objects
import os
import numpy as np
import pickle
import tifffile as tf
import sys

sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
import alloptical_utils_pj as ao


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


def run_photostim_processing(trial, exp_type, tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc, pkl_path, metainfo,
                             new_tiffs, matlab_badframes_path=None, processed_tiffs=True, discard_all=False,
                             analysis_save_path=''):

    print('\n-----Processing trial # %s-----' % trial)

    paths = [[tiffs_loc_dir, tiffs_loc, naparms_loc, paqs_loc]]
    # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

    if 'post' in exp_type and '4ap' in exp_type:
        expobj = ao.Post4ap(paths[0], metainfo=metainfo, stimtype='2pstim')
    elif 'pre' in exp_type and '4ap' in exp_type:
        expobj = ao.alloptical(paths[0], metainfo=metainfo, stimtype='2pstim')
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

    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("\nPkl saved to %s" % pkl_path)

    # collect information about seizures
    if 'post' in exp_type and '4ap' in exp_type:
        expobj.collect_seizures_info(seizures_info_array=matlab_badframes_path, discard_all=discard_all)

    if len(expobj.bad_frames) > 0:
        print('***  Collected a total of ', len(expobj.bad_frames),
              'photostim + seizure/CSD frames +  additional bad frames to bad_frames.npy  ***')

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

    print('\n----- COMPLETED RUNNING run_photostim_processing() *******')
    print(metainfo)


# %% update the trial and photostim experiment files information below before running run_photostim_processing()
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-10'
animal_prep = 'PS06'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
date = '2021-01-10'
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 5 things for every trial
trial = 't-016'  # note that %s magic command in the code below will be using these trials listed here
naparms_loc = '/photostim/ 2021-01-10_PS06_photostim_012/'  # make sure to include '/' at the end to indicate the child directory
exp_type = 'post 4ap 2p all optical'  # use 'post' and '4ap' in the description to create the appropriate post4ap exp object
comments = '10 cells x 5 groups; 7mW per cell preset: 2021-01-07_PS_250ms-stim-40hz-multi.mat (prot. #2b); not really getting seizures anymore'
paqs_loc = '%s/%s_PS06_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials
# paqs_loc = '%s/%s_RL111_%s.paq' % (data_path_base, date, '008')  # path to the .paq files for the selected trials
######



tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

# matlab_badframes_path = '%s/paired_measurements/2020-12-20_RL111_%s.mat' % (analysis_save_path[:-17], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
matlab_badframes_path = None

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

run_photostim_processing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs, metainfo=metainfo,
                         tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc, naparms_loc=(data_path_base+naparms_loc),
                         paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                         processed_tiffs=False, discard_all=True, analysis_save_path=analysis_save_path)


# %% MAKING A BIG bad_frames.npy FILE FOR ALL TRIALS STITCHED TOGETHER (RUN THIS BEFORE RUNNING SUITE2P FOR ALL OPTICAL EXPERIMENTS)

## the code below is run as part of the jupyter notebooks for each experiment's suite2p run
cont = True
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