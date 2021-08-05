import utils.funcs_pj as pj
from utils.paq_utils import paq_read, frames_discard
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt
import numpy as np

import tifffile as tf

# %%
prep = 'PS18'
date = '2021-02-02'
trials = ['t-002', 't-004', 't-005', 't-006', 't-007', 't-008', 't-009']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
    pj.plot_single_tiff(expobj.tiff_path, frame_num=0, title='%s - frame# %s' % trial)

    # cropped_tiff = aoutils.subselect_tiff(expobj.tiff_path, select_frames=(2668, 4471))#, save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr.tif' % (date, prep, date, trial, date, trial))
    aoutils.SaveDownsampledTiff(tiff_path=expobj.tiff_path, #stack=cropped_tiff,
                                save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr_downsampled.tif' % (date, prep, date, trial, date, trial))
    # expobj.collect_seizures_info(seizures_lfp_timing_matarray='/home/pshah/mnt/qnap/Analysis/%s/%s/paired_measurements/%s_%s_%s.mat' % (date, prep, date, prep, trial[-3:]))
    # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=False, save_img=True, force_redo=True)

    # expobj.MeanSeizureImages(
    #     baseline_tiff="/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_t-005/2020-12-18_t-005_Cycle00001_Ch3.tif",
    #     frames_last=1000)




# %% checking number of frames in paq file and the 2p tiff path

prep = 'PS18'
date = '2021-02-02'
trials = ['t-002', 't-004', 't-005', 't-006', 't-007', 't-008', 't-009']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
    pj.plot_single_tiff(expobj.tiff_path, frame_num=0, title='%s - frame 0' % trial)
    # open tiff file
    print('\nWorking on... %s' % expobj.tiff_path)
    tiffstack = tf.imread(expobj.tiff_path)
    n_frames_tiff = len(tiffstack)
    if not hasattr(expobj, 'frame_clock_actual'):
        expobj.paqProcessing()
    n_frames_paq = len(expobj.frame_clock_actual)

    print('|- n_frames_tiff: %s      n_frames_paq: %s' % (n_frames_tiff, n_frames_paq))



# %%

prep = 'PS06'
date = '2021-01-10'
trials = ['t-008', 't-009', 't-010', 't-011']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)

    print('\n%s' % trial)
    print('frame clock count: ', len(expobj.frame_clock_actual))
    print('raw Flu trace count: ', len(expobj.meanRawFluTrace))
    print('xml nframes', expobj.n_frames)

# paq_read(expobj.paq_path, plot=True)


# %%
data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
animal_prep = 'RL109'
date = '2020-12-19'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

trials = ['t-007']
for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    # trial = 't-009'  # note that %s magic command in the code below will be using these trials listed here
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)

    # expobj._findTargets()
    # expobj.save()

    # aoplot.plotSLMtargetsLocs(expobj, background=expobj.meanFluImg_registered, title=None)

    aoutils.slm_targets_responses(expobj, experiment, trial, y_spacing_factor=4, smooth_overlap_traces=5, figsize=[30, 20],
                                  linewidth_overlap_traces=0.2, y_lims_periphotostim_trace=[-0.5, 3.0], v_lims_periphotostim_heatmap=[-0.5, 1.0],
                                  save_results=False)




# %% re-running pre-processing on an expobj that already exists for a given trial

data_path_base = '/home/pshah/mnt/qnap/Data/2021-02-02'
animal_prep = 'PS17'
date = data_path_base[-10:]
for trial in ['t-009', 't-011']:
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
    expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
    comments = expobj.metainfo['comments']
    naparms_loc = expobj.naparm_path[-41:]
    exp_type = expobj.metainfo['exptype']
    analysis_save_path = '/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, animal_prep)

    paqs_loc = '%s/%s_%s_%s.paq' % (
    data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
    tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
    analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]
    tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
    pkl_path = "%s/%s_%s.pkl" % (analysis_save_path, date, trial)  # specify path in Analysis folder to save pkl object
    # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
    new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
    matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-17], date, animal_prep, trial[
                                                                                                                  2:])  # choose matlab path if need to use or use None for no additional bad frames
    metainfo = expobj.metainfo

    expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs,
                                                 metainfo=metainfo,
                                                 tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc,
                                                 naparms_loc=(data_path_base + naparms_loc),
                                                 paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
                                                 processed_tiffs=False, discard_all=True,
                                                 analysis_save_path=analysis_save_path)


# %%
pkl_path = '/home/pshah/mnt/qnap/Analysis/2021-01-08/PS05/2021-01-08_t-011/2021-01-08_t-011.pkl'
expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)


# x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])

from matplotlib.colors import ColorConverter
c = ColorConverter().to_rgb
bwr_custom = pj.make_colormap([c('blue'), c('white'), 0.28, c('white'), c('red')])
aoplot.plot_traces_heatmap(expobj.SLMTargets_stims_dfstdF_avg, expobj, vmin=-2, vmax=5, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, x_axis='time', cbar=True,
                           title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] + ' - targets only'), xlims=(0, expobj.pre_stim +expobj.stim_duration_frames+ expobj.post_stim), cmap=bwr_custom)






# expobj.stim_start_frames = expobj.stim_start_frames + 3
# aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
#
#
# fig, ax = plt.subplots(figsize=[20, 3])


#
#
# #%%
# # paq_path = '/home/pshah/mnt/qnap/Data/2021-01-19/2021-01-19_PS07_015.paq'
# paq, _ = paq_read(expobj.paq_path, plot=True)
#
#%%

# CREATE AND SAVE DOWNSAMPLED TIFF
prep = 'PS06'
date = '2021-01-10'
trials = ['t-008']

for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
    stack = pj.subselect_tiff(
        tiff_path=expobj.tiff_path, select_frames=(-500, -1))
    # stack = pj.subselect_tiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
    #                           select_frames=(-2000, -1))
    pj.SaveDownsampledTiff(stack=stack, save_as="/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_Cycle00001_Ch3_cropped_downsampled1.tif" % (date, prep, date, trial, date, trial))



#%% PLOT THE ZPROFILE OF A TIFF STACK
trial = 't-015'
date = '2021-01-19'

pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
            plot_image=True, figsize=[20, 4], title=(date + trial))
