import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt
import numpy as np


# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)


# i = 'pre'
# key = 'h'
# j = 0
#
# # import expobj
# expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')
#
# expobj, experiment = aoutils.import_expobj(prep='PS11', trial='t-010')
#
# expobj._findTargetsAreas()
# expobj._findTargetedS2pROIs(force_redo=True, plot=False)

# %% updating the non-targets exclusion region to 15um from the center of the spiral coordinate

ls = ['pre', 'post']
for i in ls:
    ncols = 5
    nrows = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 30))
    fig.tight_layout()
    counter = 0

    for key in list(allopticalResults.trial_maps[i].keys()):
        for j in range(len(allopticalResults.trial_maps[i][key])):
            # import expobj
            expobj, experiment = aoutils.import_expobj(aoresults_map_id=f'{i} {key}.{j}')

            expobj._parseNAPARMgpl()
            expobj._findTargetsAreas()
            expobj._findTargetedS2pROIs(force_redo=False, plot=False)

            ax = axs[counter // ncols, counter % ncols]

            fig, ax = aoplot.plot_cells_loc(expobj, cells=expobj.s2p_cell_targets, show=False, fig=fig, ax=ax,
                                            show_s2p_targets=True, title=None, invert_y=True)

            targ_img = np.zeros([expobj.frame_x, expobj.frame_y], dtype='uint16')
            target_areas = np.array(expobj.target_areas)
            targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
            ax.imshow(targ_img, cmap='Greys_r', zorder=0)
            ax.set_title(f"{expobj.metainfo['animal prep.']}, {expobj.metainfo['trial']}")

            counter += 1

    title = f"s2p cell targets (red-filled) and target areas (white) - {i}4ap trials"
    save_path = f'/home/pshah/mnt/qnap/Analysis/Results_figs/{title}.png'
    plt.suptitle(title, y=0.90)
    plt.savefig(save_path)
    # fig.show()

#
# self._findTargetedCells()
#
#
# listdir = os.listdir(self.naparm_path)
# scale_factor = self.frame_x / 512
# ## All SLM targets
# for path in listdir:
#     if 'AllFOVTargets' in path:
#         target_file = path
# target_image = tf.imread(os.path.join(self.naparm_path, target_file))
#
# plt.imshow(target_image, cmap='Greys')
# plt.show()
#
#
#
#
# targetsareasFOV = np.zeros((self.frame_x, self.frame_y))
# for (y,x) in pj.flattenOnce(self.target_areas):
#     targetsareasFOV[x, y] = 1
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(targetsareasFOV, cmap='Greys_r', zorder=0)
# ax.set_title('self.target_areas')
# fig.show()
#
#
#
# targ_img = np.zeros([self.frame_x, self.frame_y], dtype='uint16')
# target_areas = np.array(self.target_areas)
# targ_img[target_areas[:, :, 1], target_areas[:, :, 0]] = 1
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(targ_img, cmap='Greys_r', zorder=0)
# ax.set_title('12')
# fig.show()
#
#
#
# fig, ax = plt.subplots(figsize=(6,6))
# ax.imshow(cell_img, cmap='Greys_r', zorder=0)
# ax.set_title('s2p cell targets (red-filled) and all target coords (green) %s/%s' % (
#     self.metainfo['trial'], self.metainfo['animal prep.']))
# fig.show()




print('yes')


# # %%
# prep = 'PS18'
# date = '2021-02-02'
# trials = ['t-002', 't-004', 't-005', 't-006', 't-007', 't-008', 't-009']
#
# for trial in trials:
#     ###### IMPORT pkl file containing data in form of expobj
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
#
#     expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
#
#     # pj.plot_single_tiff(expobj.tiff_path, frame_num=201, title='%s - frame# 201' % trial)
#     # cropped_tiff = aoutils.subselect_tiff(expobj.tiff_path, select_frames=(2668, 4471))#, save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr.tif' % (date, prep, date, trial, date, trial))
#     # aoutils.SaveDownsampledTiff(tiff_path=expobj.tiff_path, #stack=cropped_tiff,
#     #                             save_as='/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_2668-4471fr_downsampled.tif' % (date, prep, date, trial, date, trial))
#     # expobj.collect_seizures_info(seizures_lfp_timing_matarray='/home/pshah/mnt/qnap/Analysis/%s/%s/paired_measurements/%s_%s_%s.mat' % (date, prep, date, prep, trial[-3:]))
#     # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=False, save_img=True, force_redo=True)
#
#     # expobj.MeanSeizureImages(
#     #     baseline_tiff="/home/pshah/mnt/qnap/Data/2020-12-18/2020-12-18_t-005/2020-12-18_t-005_Cycle00001_Ch3.tif",
#     #     frames_last=1000)
#
#
# # %% add 100ms to the stim dur for expobj which need it (i.e. trials where the stim end is just after the stim_dur and traces are still coming down)
# ls = ['PS05 t-010', 'PS06 t-011', 'PS11 t-010', 'PS17 t-005', 'PS17 t-006', 'PS17 t-007', 'PS18 t-006']
# for i in ls:
#     prep, trial = re.split(' ', i)
#     expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
#     expobj.stim_dur = expobj.stim_dur + 100
#     expobj.save()
#
# # %% plot signals of suite2p outputs of a cell with F-neuropil and neuropil -- trying to see if neuropil signal contains anything of predictive value for the cell's spiking activity?
#
# i = allopticalResults.post_4ap_trials[0]
# j = 0
# prep = i[j][:-6]
# trial = i[j][-5:]
# print('\nLoading up... ', prep, trial)
# expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)
#
# cell = 10
# # window = [0, expobj.n_frames]
# fig, ax = plt.subplots(figsize=(10, 3))
# ax2 = ax.twinx()
# ax.plot(expobj.frame_clock_actual[:expobj.n_frames], expobj.raw[cell], color='black', lw=0.2)
# # ax.plot(expobj.spks[cell], color='blue')
# ax.plot(expobj.frame_clock_actual[:expobj.n_frames], expobj.neuropil[cell], color='red', lw=0.2)
# ax2.plot(expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual], color='steelblue', lw=0.2)
# # ax.set_xlim(window[0], window[1])
# fig.show()
#
#
#
#
# # %% running processing of SLM targets responses outsz
#
# prep = 'RL108'
# trial = 't-013'
# date = ls(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == (prep + ' ' + trial), 'date'])[0]
#
# expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep, verbose=False)
# hasattr(expobj, 'outsz_responses_SLMtargets')
#
#
# # %% fixing squashing of images issue in PS18 trials
#
# import cv2
#
#
# prep = 'PS18'
# date = '2021-02-02'
# trial = 't-006'
#
# ###### IMPORT pkl file containing data in form of expobj
# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
# expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
# fr = pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)
#
# # unsquash the bottom 2/3rd of the frame
# lw = fr.shape[0]
# fr_ = fr[int(1/3*lw):, ]
# res = cv2.resize(fr_, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
#
# # plot image
# plt.imshow(res, cmap='gray')
# plt.suptitle('%s' % trial)
# plt.show()
#
#
#
# # compare with spont trial that was correct
# trial = 't-002'
#
# ###### IMPORT pkl file containing data in form of expobj
# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
# expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
# fr = pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)
#
# # plot image
# plt.imshow(fr, cmap='gray')
# plt.suptitle('%s' % trial)
# plt.show()
#
#
#
#
# # open tiff file
# print('\nWorking on... %s' % expobj.tiff_path)
# tiffstack = tf.imread(expobj.tiff_path)
#
#
#
# # %% checking number of frames in paq file and the 2p tiff path
#
# prep = 'PS18'
# date = '2021-02-02'
# trials = ['t-008', 't-009']
#
# for trial in trials:
#     ###### IMPORT pkl file containing data in form of expobj
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
#
#     expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
#     pj.plot_single_tiff(expobj.tiff_path, frame_num=238, title='%s - frame 238' % trial)
#     # open tiff file
#     print('\nWorking on... %s' % expobj.tiff_path)
#     tiffstack = tf.imread(expobj.tiff_path)
#     n_frames_tiff = len(tiffstack)
#     if not hasattr(expobj, 'frame_clock_actual'):
#         expobj.paqProcessing()
#     n_frames_paq = len(expobj.frame_clock_actual)
#
#     print('|- n_frames_tiff: %s      n_frames_paq: %s' % (n_frames_tiff, n_frames_paq))
#
#
#
# # %%
#
# prep = 'PS06'
# date = '2021-01-10'
# trials = ['t-008', 't-009', 't-010', 't-011']
#
# for trial in trials:
#     ###### IMPORT pkl file containing data in form of expobj
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
#
#     expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
#
#     print('\n%s' % trial)
#     print('frame clock count: ', len(expobj.frame_clock_actual))
#     print('raw Flu trace count: ', len(expobj.meanRawFluTrace))
#     print('xml nframes', expobj.n_frames)
#
# # paq_read(expobj.paq_path, plot=True)
#
#
# # %%
# data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
# animal_prep = 'RL109'
# date = '2020-12-19'
# # specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# # paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials
#
# trials = ['t-007']
# for trial in trials:
#     ###### IMPORT pkl file containing data in form of expobj
#     # trial = 't-009'  # note that %s magic command in the code below will be using these trials listed here
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/RL109/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
#     expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
#
#     # expobj._findTargets()
#     # expobj.save()
#
#     # aoplot.plotSLMtargetsLocs(expobj, background=expobj.meanFluImg_registered, title=None)
#
#     aoutils.slm_targets_responses(expobj, experiment, trial, y_spacing_factor=4, smooth_overlap_traces=5, figsize=[30, 20],
#                                   linewidth_overlap_traces=0.2, y_lims_periphotostim_trace=[-0.5, 3.0], v_lims_periphotostim_heatmap=[-0.5, 1.0],
#                                   save_results=False)
#
#
#
#
# # %% re-running pre-processing on an expobj that already exists for a given trial
#
# data_path_base = '/home/pshah/mnt/qnap/Data/2021-02-02'
# animal_prep = 'PS17'
# date = data_path_base[-10:]
# for trial in ['t-009', 't-011']:
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
#     expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
#     comments = expobj.metainfo['comments']
#     naparms_loc = expobj.naparm_path[-41:]
#     exp_type = expobj.metainfo['exptype']
#     analysis_save_path = '/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, animal_prep)
#
#     paqs_loc = '%s/%s_%s_%s.paq' % (
#     data_path_base, date, animal_prep, trial[2:])  # path to the .paq files for the selected trials
#     tiffs_loc_dir = '%s/%s_%s' % (data_path_base, date, trial)
#     analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]
#     tiffs_loc = '%s/%s_%s_Cycle00001_Ch3.tif' % (tiffs_loc_dir, date, trial)
#     pkl_path = "%s/%s_%s.pkl" % (analysis_save_path, date, trial)  # specify path in Analysis folder to save pkl object
#     # paqs_loc = '%s/%s_RL109_010.paq' % (data_path_base, date)  # path to the .paq files for the selected trials
#     new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved
#     matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:-17], date, animal_prep, trial[
#                                                                                                                   2:])  # choose matlab path if need to use or use None for no additional bad frames
#     metainfo = expobj.metainfo
#
#     expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs,
#                                                  metainfo=metainfo,
#                                                  tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc,
#                                                  naparms_loc=(data_path_base + naparms_loc),
#                                                  paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path,
#                                                  processed_tiffs=False, discard_all=True,
#                                                  analysis_save_path=analysis_save_path)
#
#
# # %%
# pkl_path = '/home/pshah/mnt/qnap/Analysis/2021-01-08/PS05/2021-01-08_t-011/2021-01-08_t-011.pkl'
# expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)
#
#
# # x = np.asarray([i for i in expobj.SLMTargets_stims_dfstdF_avg])
#
# from matplotlib.colors import ColorConverter
# c = ColorConverter().to_rgb
# bwr_custom = pj.make_colormap([c('blue'), c('white'), 0.28, c('white'), c('red')])
# aoplot.plot_traces_heatmap(expobj.SLMTargets_stims_dfstdF_avg, expobj, vmin=-2, vmax=5, stim_on=expobj.pre_stim, stim_off=expobj.pre_stim + expobj.stim_duration_frames + 1, x_axis='time', cbar=True,
#                            title=(expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] + ' - targets only'), xlims=(0, expobj.pre_stim +expobj.stim_duration_frames+ expobj.post_stim), cmap=bwr_custom)
#
#
#
#
#
#
# # expobj.stim_start_frames = expobj.stim_start_frames + 3
# # aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, x_axis='frames', figsize=[20, 3])
# #
# #
# # fig, ax = plt.subplots(figsize=[20, 3])
#
#
# #
# #
# # #%%
# # # paq_path = '/home/pshah/mnt/qnap/Data/2021-01-19/2021-01-19_PS07_015.paq'
# # paq, _ = paq_read(expobj.paq_path, plot=True)
# #
# #%%
#
# # CREATE AND SAVE DOWNSAMPLED TIFF
# prep = 'PS06'
# date = '2021-01-10'
# trials = ['t-008']
#
# for trial in trials:
#     ###### IMPORT pkl file containing data in form of expobj
#     pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
#
#     expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
#     stack = pj.subselect_tiff(
#         tiff_path=expobj.tiff_path, select_frames=(-500, -1))
#     # stack = pj.subselect_tiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
#     #                           select_frames=(-2000, -1))
#     pj.SaveDownsampledTiff(stack=stack, save_as="/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s_Cycle00001_Ch3_cropped_downsampled1.tif" % (date, prep, date, trial, date, trial))
#
#
#
# #%% PLOT THE ZPROFILE OF A TIFF STACK
# trial = 't-015'
# date = '2021-01-19'
#
# pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
#             plot_image=True, figsize=[20, 4], title=(date + trial))
#
#
# # %% PLOT LFP WITH STIM TIMINGS FOR ALL-OPTICAL EXPERIMENT
#
# prep = 'PS07'
# trial = 't-017'
# date = ls(allopticalResults.metainfo.loc[allopticalResults.metainfo['prep_trial'] == (prep + ' ' + trial), 'date'])[0]
#
# expobj, experiment = aoutils.import_expobj(trial=trial, date=date, prep=prep)
# aoplot.plot_lfp_stims(expobj)
