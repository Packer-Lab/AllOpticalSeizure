import alloptical_utils_pj as aoutils
import utils.funcs_pj
import utils.funcs_pj as pj

#%% CREATE AND SAVE DOWNSAMPLED TIFF
trial = 't-010'
date = '2021-02-01'
utils.funcs_pj.SaveDownsampledTiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial))


# to download downsampled: -01-10 t-014 (look for seizure, compared to paq); 01-08 t-012,

#%% PLOT THE ZPROFILE OF A TIFF STACK
trial = 't-015'
date = '2021-01-19'

pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
            plot_image=True, figsize=[20, 4], title=(date + trial))
