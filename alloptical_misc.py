import utils.funcs_pj as pj

#%% CREATE AND SAVE DOWNSAMPLED TIFF
trial = 't-011'
date = '2021-01-11'
pj.SaveDownsampledTiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial))

#%% PLOT THE ZPROFILE OF A TIFF STACK
trial = 't-015'
date = '2021-01-19'

pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
            plot_image=True, figsize=[20, 4], title=(date + trial))
