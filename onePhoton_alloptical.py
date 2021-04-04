### code for analysis of one photon photostim experiments

## starting with PS01

# from utils.paq_utils import paq_read
# import utils.funcs_pj as pjf
# import matplotlib.pyplot as plt
# import numpy as np
import os
import alloptical_utils_pj as aoutils

# %%
data_path_base = '/home/pshah/mnt/qnap/Data/2021-01-06'
animal_prep = 'PS001'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
date = data_path_base[-10:]
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

# need to update these 5 things for every trial
trial = 't-004'  # note that %s magic command in the code below will be using these trials listed here
exp_type = '1p photostim'
comments = '2x 1p opto stim; tiff images are built properly'
paqs_loc = '%s/%s_PS001_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials


tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
# paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
# make the necessary Analysis saving subfolder as well
analysis_save_path = tiffs_loc[:21] + 'Analysis/' + tiffs_loc_dir[26:]

metainfo = {
    'animal prep.': animal_prep,
    'trial': trial,
    'date': date,
    'exptype': exp_type,
    'data_path_base': data_path_base,
    'comments': comments
}

def run_1p_processing(tiffs_loc_dir, tiffs_loc, paqs_loc, pkl_path, metainfo):
    print('\n-----Processing trial # %s-----' % trial)

    paths = [tiffs_loc_dir, tiffs_loc, paqs_loc]
    # print('tiffs_loc_dir, naparms_loc, paqs_loc paths:\n', paths)

    expobj = aoutils.onePstim(paths, metainfo)

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

    expobj.save_pkl(pkl_path=pkl_path)

    return expobj

expobj = run_1p_processing(tiffs_loc_dir, tiffs_loc, paqs_loc, pkl_path, metainfo)


# %%
###### IMPORT pkl file containing data in form of expobj
trial = 't-004'
date = '2021-01-06'
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
# pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)

expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path)


# %% # look at the average Ca Flu trace pre and post stim, just calculate the average of the whole frame and plot as continuous timeseries
# - this approach should also allow to look at the stims that give rise to extended seizure events where the Ca Flu stays up

import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np

def analyze_flu_trace_1pstim(expobj):
    pass

### make a new tiff file (not for suite2p) with the first photostim frame whitened, and save new tiff
print('\n-----making processed photostim .tiff from:')
tiff_path = tiffs_loc
print(tiff_path)
im_stack = tf.imread(tiff_path, key=range(expobj.n_frames))
print('Processing experiment tiff of shape: ', im_stack.shape)

im_avg = np.mean(np.mean(im_stack, axis=1), axis=1); print(im_avg.shape)


# make plot of avg Ca trace
plt.figure(figsize=[10,3])
plt.plot(im_avg, c='seagreen')
plt.show()


#%%
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