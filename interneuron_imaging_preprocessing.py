### code for preprocessing (initiating expobj classes) of interneuron experiments

from funcsforprajay import funcs as pj
import alloptical_utils_pj as aoutils

# %%
prep = 'PS15'
date = '2021-01-29'

# need to update these 3 things for every trial
# trial = 't-012'  # note that %s magic command in the code below will be using these trials listed here
trials = ['t-001', 't-002', 't-003']  # note that %s magic command in the code below will be using these trials listed here
exp_type = 'interneuron imaging, pre 4ap'
comments = ['interneuron imaging - problems with leftover raw/single tiffs during backup.py converting to tiffs - need to confirm that all data is in tiff format correctly, there are some leftover RAW files in the computer on G: as well']

for trial in trials:
    metainfo = {
        'animal prep.': prep,
        'trial': trial,
        'date': date,
        'exptype': exp_type,
        'data_path_base': '/home/pshah/mnt/qnap/Data/%s_old_copy/' % date,
        'comments': comments
    }
    expobj = aoutils.TwoPhotonImaging(
        tiff_path=metainfo['data_path_base'] + '%s_%s/%s_%s_Cycle00001_Ch3.tif' % (date, trial, date, trial),
        paq_path=metainfo['data_path_base'] + '%s_%s_%s.paq' % (date, prep, trial[-3:]), metainfo=metainfo,
        analysis_save_path='/home/pshah/mnt/qnap/Analysis/%s/%s/' % (date, prep), quick=False)
    expobj.paqProcessing()
    expobj.save()
    print('tiff path: ', expobj.tiff_path)
    print('\nn_frames: ', expobj.n_frames)
    print('n_frames_tiff: ', expobj.meanRawFluTrace.shape)
    print('n_frame_clock_paq: ', len(expobj.frame_clock_actual))

# NOTE: NEED TO RERUN FOR PS15 WITH THE CORRECT TIFF FILES AND STUFF

# %%
# import experiment object and look for number

date = '2021-01-29'
prep = 'PS15'
trials = ['t-001', 't-002', 't-003', 't-005', 't-004', 't-006']

for trial in trials:
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s.pkl" % (date, prep, date, trial)

    expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)
    # expobj.mean_raw_flu_trace(plot=True)
    # aoplot.plotMeanRawFluTrace(expobj=expobj, stim_span_color=None, stim_lines=False, x_axis='frames', figsize=[20, 3],
    #                                        title='Mean raw Flu trace -')
    if len(expobj.tiff_path) == 2:
        pj.plot_single_tiff(expobj.tiff_path[1], frame_num=201, title='%s - frame# 201' % trial)
        print(expobj.tiff_path[1])
    else:
        pj.plot_single_tiff(expobj.tiff_path, frame_num=201, title='%s - frame# 201' % trial)
        print(expobj.tiff_path)

    print('\nn_frames: ', expobj.n_frames)
    print('n_frames_tiff: ', expobj.meanRawFluTrace.shape)
    print('n_frame_clock_paq: ', len(expobj.frame_clock_actual))