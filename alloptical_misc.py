import utils.funcs_pj as pj
from utils.paq_utils import paq_read, frames_discard
import alloptical_utils_pj as aoutils

###### IMPORT pkl file containing data in form of expobj
trial = 't-012'
date = '2021-01-09'

expobj, experiment = aoutils.import_expobj(trial=trial, date=date)


#%%
# paq_path = '/home/pshah/mnt/qnap/Data/2021-01-19/2021-01-19_PS07_015.paq'
paq, _ = paq_read(expobj.paq_path, plot=True)

#%%

# CREATE AND SAVE DOWNSAMPLED TIFF
trial = 't-006'
date = '2021-01-08'

stack = pj.subselect_tiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
                          select_frames=(-2000, -1))

# pj.SaveDownsampledTiff(tiff_path="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial))
pj.SaveDownsampledTiff(stack=stack, save_as="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3_cropped_downsampled_2.tif" % (date, date, trial, date, trial))



#%% PLOT THE ZPROFILE OF A TIFF STACK
trial = 't-015'
date = '2021-01-19'

pj.ZProfile(movie="/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s_Cycle00001_Ch3.tif" % (date, date, trial, date, trial),
            plot_image=True, figsize=[20, 4], title=(date + trial))
