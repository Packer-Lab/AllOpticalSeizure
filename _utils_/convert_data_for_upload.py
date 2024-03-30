# %% convert individual datasets to upload format
from typing import Union

import numpy as np
import pandas as pd

from _utils_._alloptical_utils import run_for_loop_across_exps
from _main_.AllOpticalMain import alloptical
from _main_.Post4apMain import Post4ap
from _utils_._anndata import AnnotatedData2
from _utils_.io import import_expobj


# from pynwb.file import Subject
# from pynwb import NWBFile, TimeSeries, NWBHDF5IO
# from pynwb.image import ImageSeries
# from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, \
#     Fluorescence, CorrectedImageStack, MotionCorrection, RoiResponseSeries


"""
LIST OF THINGS TO INCLUDE IN DATASHARE:

FOR EACH INDIVIDUAL ANIMAL (N = 6):
- 1x suite2p expobj.stat, and ops.npy from suite2p output 
- 1x pre-4ap trial processed data in anndata format:
- 1x post-4ap trial processed data in anndata format

anndata objects of:
    primary datamatrix of cells x frames:
    - X = raw data (from suite2p)
    - layers = normalized data
    - var:
        - lfp
        - stim timing frames
        - seizure wavefront location
    - obs:
        - SLM target coordinates
    - obsm:
        - SLM target areas 
    
    photostim response processing:
        1) SLM targets dFF responses [PhotostimAnalysisSlmTargets.X]
        2) Non targets dFF responses [PhotostimResponsesNonTargets.X]
    


"""



def load_exp(exp_prep):
    prep = exp_prep[:-6]
    trial = exp_prep[-5:]
    try:
        expobj = import_expobj(exp_prep=exp_prep)
    except:
        raise ImportError(f"IMPORT ERROR IN {prep} {trial}")
    return expobj

@run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=0, run_trials=['RL108 t-013'])
def convert_dataset(**kwargs):
    # LOAD TRIAL...
    expobj: Union[alloptical, Post4ap] = kwargs['expobj']

    # CONVERT TRIAL TO COMPREHENSIBLE DATASET INCLUDING SUITE2P OUTPUT, ANNOTATED DATA,
    obs = pd.DataFrame(data=[(stat['med'][1], stat['med'][0]) for i, stat in enumerate(expobj.stat)],
                       columns=['s2p cell - x coord', 's2p cell - y coord'], index=range(expobj.raw.shape[0]))
    var_meta = pd.DataFrame(columns=['LFP (downsampled)', 'Photostimulation ON', 'Seizure wavefront location'],
                            index=range(expobj.raw.shape[1]))
    var_meta['Photostimulation ON'] = [False] * expobj.n_frames
    for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
        var_meta['Photostimulation ON'][stim_frame: stim_frame + expobj.stim_duration_frames + 1] = True
    var_meta['LFP (downsampled)'] = expobj.lfp_signal[expobj.frame_clock_actual[:expobj.n_frames]]
    var_meta['Seizure wavefront location'] = None
    for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
        for i in range(stim_frame, stim_frame + expobj.stim_duration_frames + 1):
            var_meta['Seizure wavefront location'][i] = tuple(
                expobj.PhotostimAnalysisSlmTargets.adata.var['seizure location'][fr_idx]) if \
            expobj.PhotostimAnalysisSlmTargets.adata.var['seizure location'][fr_idx] is not None else None

    expobj.primary_adata = AnnotatedData2(X=expobj.raw, obs=obs, var=var_meta,
                                   data_label='Primary imaging datamatrix (all cells)')

    ###
    obs = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord'])
    var = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.adata.var['seizure location'])
    if 'post' not in expobj.exptype:
        var = pd.DataFrame(data=[None] * expobj.PhotostimResponsesSLMTargets.adata.shape[1],
                           columns=['seizure location'], index=expobj.PhotostimResponsesSLMTargets.adata.shape[1])
    expobj.target_adata = AnnotatedData2(X=expobj.PhotostimResponsesSLMTargets.adata.X,
                                  obs=obs, var=var, data_label='SLM Targets')


    ###
    obs = pd.DataFrame(expobj.PhotostimResponsesNonTargets.adata.obs['distance to nearest target (um)'])
    var = pd.DataFrame(expobj.PhotostimResponsesNonTargets.adata.var['seizure location'])
    if 'post' not in expobj.exptype:
        var = pd.DataFrame(data=[None] * expobj.PhotostimResponsesNonTargets.adata.shape[1],
                           columns=['seizure location'], index=expobj.PhotostimResponsesNonTargets.adata.shape[1])
    expobj.nontarget_adata = AnnotatedData2(X=expobj.PhotostimResponsesNonTargets.adata.X,
                                     obs=obs, var=var, data_label='Non Targets')

    return expobj





expobj: Post4ap = load_exp(exp_prep='RL108 t-013')


expobj.metainfo  # metainfo about one animal


## SUITE2P DATA FOR ANIMAL
expobj.s2p_path  # s2p path
# or
expobj.suite2p_path  # s2p path might also be here...



# %% s2p image trace

frame = np.array([[100]*expobj.frame_x]*expobj.frame_y)
for i, stat in enumerate(expobj.stat):
    frame[stat['ypix'], stat['xpix']] = 0

import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig, ax = plt.subplots(figsize=(5,5), dpi=300)
ax.imshow(frame, cmap=cm.gray)
ax.set_axis_off()
fig.show()

# %% new anndata


"""
    primary datamatrix of cells x frames:
    - X = raw data (from suite2p)
    - var:
        - lfp
        - stim timing frames
        - seizure wavefront location
    - obs:
        - s2p cell coord

"""


obs = pd.DataFrame(data=[(stat['med'][1], stat['med'][0]) for i, stat in enumerate(expobj.stat)], columns=['s2p cell - x coord', 's2p cell - y coord'], index=range(expobj.raw.shape[0]))
var_meta = pd.DataFrame(columns=['LFP (downsampled)', 'Photostimulation ON', 'Seizure wavefront location'], index=range(expobj.raw.shape[1]))
var_meta['Photostimulation ON'] = [False] * expobj.n_frames
for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
    var_meta['Photostimulation ON'][stim_frame: stim_frame + expobj.stim_duration_frames + 1] = True
var_meta['LFP (downsampled)'] = expobj.lfp_signal[expobj.frame_clock_actual[:expobj.n_frames]]
var_meta['Seizure wavefront location'] = None
for fr_idx, stim_frame in enumerate(expobj.stim_start_frames):
    for i in range(stim_frame, stim_frame + expobj.stim_duration_frames + 1):
        var_meta['Seizure wavefront location'][i] = tuple(expobj.PhotostimAnalysisSlmTargets.adata.var['seizure location'][fr_idx]) if expobj.PhotostimAnalysisSlmTargets.adata.var['seizure location'][fr_idx] is not None else None

primary_adata = AnnotatedData2(X=expobj.raw, obs=obs, var=var_meta, data_label='Primary imaging datamatrix (all cells)')


# %%

"""
      photostim response processing:
        1) SLM targets dFF responses [PhotostimAnalysisSlmTargets.X]
        2) Non targets dFF responses [PhotostimResponsesNonTargets.X]
    
"""

obs = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.adata.obs['SLM target coord'])
var = pd.DataFrame(expobj.PhotostimResponsesSLMTargets.adata.var['seizure location'])
if 'post' not in expobj.exptype:
    var = pd.DataFrame(data=[None]*expobj.PhotostimResponsesSLMTargets.adata.shape[1], columns=['seizure location'], index=expobj.PhotostimResponsesSLMTargets.adata.shape[1])
target_adata = AnnotatedData2(X=expobj.PhotostimResponsesSLMTargets.adata.X,
                              obs=obs, var=var, data_label='SLM Targets')

obs = pd.DataFrame(expobj.PhotostimResponsesNonTargets.adata.obs['distance to nearest target (um)'])
var = pd.DataFrame(expobj.PhotostimResponsesNonTargets.adata.var['seizure location'])
if 'post' not in expobj.exptype:
    var = pd.DataFrame(data=[None]*expobj.PhotostimResponsesNonTargets.adata.shape[1], columns=['seizure location'], index=expobj.PhotostimResponsesNonTargets.adata.shape[1])
nontarget_adata = AnnotatedData2(X=expobj.PhotostimResponsesNonTargets.adata.X,
                              obs=obs, var=var, data_label='Non Targets')



# %%



def newImagingNWB(expobj: Union[alloptical, Post4ap], save=True,
                 **kwargs):

    # initialize the NWB file
    nwbfile = NWBFile(
        session_description=expobj.comment,
        identifier=expobj.trial,
        subject=expobj.prep
    )

    device = nwbfile.create_device(
        name='Bruker 2pPlus'
    )


    imaging_plane = nwbfile.create_imaging_plane(
        name='Plane 0',
        optical_channel='GCaMP',
        imaging_rate=expobj.fps,
        device=device,
        grid_spacing=[expobj.pix_sz_x, expobj.pix_sz_y],  # spacing between pixels
        grid_spacing_unit='microns per pixel',
    )

    print(f"[NWB processing]: Adding 2photon imaging series to nwb file ...")

    # add Suite2p pre-processed data to nwb file
    ophys_module = nwbfile.create_processing_module(name='Pre-processed imaging data',
                                                 description='Pre-processed imaging data (ROIs and Fluorescence)')

    # create image segmentation object
    img_seg = ImageSegmentation()

    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description=f'Image segmentation output from Plane 0',
        imaging_plane=imaging_plane,
    )
    ophys_module.add(img_seg)

    # ADD ROIs from Suite2p ROI segmentation
    for i, roi in enumerate(expobj.stat):
        pixel_mask = []
        for iy in roi['ypix'][i]:
            for ix in roi['xpix'][i]:
                pixel_mask.append((iy, ix, 1))
        ps.add_roi(pixel_mask=pixel_mask)

    # add pre-processed data to nwb file
    rt_region = ps.create_roi_table_region(
        region=list(np.arange(0, expobj.n_units)),
        description=f'Suite2p Segmented ROIs'
    )

    roi_resp_series = RoiResponseSeries(
        name='ROIs x extracted fluorescence signal',
        data=expobj.raw,
        rois=rt_region,
        rate=expobj.fps,
        unit='A.U. (GCaMP fluorescence)'
    )

    fl = Fluorescence(roi_response_series=roi_resp_series)
    ophys_module.add(fl)


    # add temporal data
    print(f"[NWB processing]: Adding temporal series to nwb file ...")
    tmseries = {
        'lfp': expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual],
        'photostimulation timed frames': expobj.stim
        - 'seizure wavefront location'
    }

    # LFP
    test_ts = TimeSeries(
        name='Local field potential (LFP)',
        data=expobj.lfp_signal[expobj.frame_start_time_actual: expobj.frame_end_time_actual],
        unit='voltage',
        timestamps=np.linspace(0, (expobj.frame_end_time_actual - expobj.frame_start_time_actual), expobj.paq_rate)
    )
    nwbfile.add_acquisition(test_ts)


    # Photostim times
    test_ts = TimeSeries(
        name='Photostimulation trials',
        data=expobj.stim_start_frames,
        unit='imaging frame #',
        timestamps=np.linspace(0, (expobj.frame_end_time_actual - expobj.frame_start_time_actual), expobj.paq_rate)
    )
    nwbfile.add_acquisition(test_ts)



    for i, channel in enumerate(tmdata.channels):
        unit = tmdata.units[i] if tmdata.units else ''

        test_ts = TimeSeries(
            name=channel,
            data=tuple(tmdata.data[channel]),
            unit=unit,
            timestamps=np.linspace(tmdata.crop_offset_time,
                                   (tmdata.n_timepoints / tmdata.sampling_rate) + tmdata.crop_offset_time,
                                   tmdata.n_timepoints)
            # account for any cropped time (from the start of the recording)
        )

        nwbfile.add_acquisition(test_ts)

    # save newly generated nwb file
    _nwb_save_path = trialobj.pkl_path[:-4] + '.nwb'

    if save:
        save_nwb(nwbfile=nwbfile, path=_nwb_save_path)
        # print(f'\n\t ** Saving nwb file to: {_nwb_save_path}')
        # os.makedirs(os.path.dirname(_nwb_save_path), exist_ok=True)
        # with NWBHDF5IO(_nwb_save_path, 'w') as io:
        #     io.write(nwbfile)

    return nwbfile


