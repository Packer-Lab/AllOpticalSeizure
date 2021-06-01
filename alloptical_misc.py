import utils.funcs_pj as pj
from utils.paq_utils import paq_read, frames_discard
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import matplotlib.pyplot as plt


data_path_base = '/home/pshah/mnt/qnap/Data/2020-12-19'
animal_prep = 'RL109'
date = '2020-12-19'
# specify location of the naparm export for the trial(s) - ensure that this export was used for all trials, if # of trials > 1
# paqs_loc = '%s/%s_RL109_%s.paq' % (data_path_base, date, trial[2:])  # path to the .paq files for the selected trials

trials = ['t-007']
for trial in trials:
    ###### IMPORT pkl file containing data in form of expobj
    # trial = 't-009'  # note that %s magic command in the code below will be using these trials listed here
    expobj, experiment = aoutils.import_expobj(trial=trial, date=date)

    aoutils.slm_targets_responses(expobj, experiment, trial, y_spacing_factor=3)


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
