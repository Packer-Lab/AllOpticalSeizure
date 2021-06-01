import utils.funcs_pj as pj
from utils.paq_utils import paq_read, frames_discard
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt


data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-18'
animal_prep = 'RL108'
# date = '2021-01-10'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

trials = ['t-012']
for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    # trial = 't-009'  # note that %s magic command in the code below will be using these trials listed here
    date = data_path_base[-10:]
    expobj, experiment = aoutils.import_expobj(trial=trial, date=date)


    # need to update these 4 things for every trial
    comments = expobj.metainfo['comments']
    naparms_loc = expobj.naparm_path
    exp_type = expobj.metainfo['exptype']

    paqs_loc = expobj.paq_path
    tiffs_loc_dir = expobj.tiff_path_dir
    tiffs_loc = expobj.tiff_path
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
    new_tiffs = tiffs_loc[:-19]  # where new tiffs from rm_artifacts_tiffs will be saved

    # make the necessary Analysis saving subfolder as well
    analysis_save_path = '/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, animal_prep)
    analysis_save_path = analysis_save_path + tiffs_loc_dir[-16:]

    # matlab_badframes_path = '%s/paired_measurements/%s_%s_%s.mat' % (analysis_save_path[:40], date, animal_prep, trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
    matlab_badframes_path = None

    metainfo = {
        'animal prep.': animal_prep,
        'trial': trial,
        'date': date,
        'exptype': exp_type,
        'data_path_base': data_path_base,
        'comments': comments
    }

    expobj = aoutils.run_photostim_preprocessing(trial, exp_type=exp_type, pkl_path=pkl_path, new_tiffs=new_tiffs, metainfo=metainfo,
                                                 tiffs_loc_dir=tiffs_loc_dir, tiffs_loc=tiffs_loc, naparms_loc=naparms_loc,
                                                 paqs_loc=paqs_loc, matlab_badframes_path=matlab_badframes_path, quick=False,
                                                 processed_tiffs=False, discard_all=True, analysis_save_path=analysis_save_path)

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
# #%%
#
# # CREATE AND SAVE DOWNSAMPLED TIFF
# trial = 't-006'
# date = '2021-01-08'
#
# stack = pj.subselect_tiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
#                           select_frames=(-2000, -1))
#
# # pj.SaveDownsampledTiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial))
# pj.SaveDownsampledTiff(stack=stack, save_as="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3_cropped_downsampled_2.tif" % (date, date, trial, date, trial))
#
#
#
# #%% PLOT THE ZPROFILE OF A TIFF STACK
# trial = 't-015'
# date = '2021-01-19'
#
# pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
#             plot_image=True, figsize=[20, 4], title=(date + trial))
