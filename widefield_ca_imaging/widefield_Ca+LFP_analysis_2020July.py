#%% ## analysis for data collected by R. Burman in July 2020 in the Packer lab
# experiment details: transgenic calcium imaging animals imaged with widefield Calcium imaging and injected with 4AP to generate focal seizures,
# also recorded with LFP

import sys; sys.path.append('/home/pshah/Documents/code/Vape/utils')
import alloptical_utils_pj as ao
from utils import funcs_pj as pjf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr
from scipy.signal import savgol_filter

import tifffile as tf
import pickle as pkl

# plt.rcParams['figure.figsize'] = [20,3]

#%%
# data is accessed from apacker's folder
# path to paq file
input_path = "/mnt/qnap_Root/apacker/Data/2020-07-17/paq/2020-07-16_HF098_003.paq"

paq = pjf.paq_read(input_path, plot=True)
camera = paq['data'][1]
voltage = paq['data'][2]

plt.figure(figsize=[10,2])
plt.plot(voltage[-18045640:], linewidth=0.2); plt.suptitle('LFP voltage'); plt.show()

#%% DOWNSAMPLING or CROPPING OF A TIFF FILE (USEFULL IF YOU WANT TO DOWNLOAD TO LOCAL COMPUTER AND WATCH THE TIFF)

# import first tiff and make a downsampled tiff for sample: HF98 - pre4ap baseline widefield Ca imaging

# open tiff file
tiff_path = '/home/pshah/mnt/qnap/Analysis/2020-07-16/HF95/wide4x4ap_010.tif'
stack = tf.imread(tiff_path)

# view a single frame from the tiff
plt.imshow(stack[101], cmap='gist_gray', vmin=0, vmax=3000); plt.show()



### Downsampling of mptiffs, and exporting to save into specified folder
i = 5
tiff_path = '/mnt/qnap_Root/apacker/Data/2020-07-17/WF/HF098/2020-07-16_HF098_00400001/2020-07-16_HF098_00400001(%s).tif' % i
save_as = '/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/004/2020-07-16_HF098_004_%s_downsampled.tif' % i
ao.SaveDownsampledTiff(tiff_path="/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/2020-07-17_HF098_003_last50pct.tif",
                       save_as="/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/2020-07-17_HF098_003_last50pct_downsampled.tif")



### subselecting frames from the tiff stack, and exporting to save into specified folder
target_frames = 500  # number of frames to cut tiff down to
stack = tf.imread('/home/pshah/mnt/qnap/Analysis/2020-07-16/HF95/wide4x4ap_011_downsampled.tif')
num_frames = stack.shape[0]
stack_cropped = stack[:target_frames]
save_as = '/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/004/2020-07-17_HF098_004_%s_subselected.tif'
tf.imwrite(file='/home/pshah/mnt/qnap/Analysis/2020-07-16/HF95/wide4x4ap_011_downsampled_cropped.tif', data=stack_cropped, photometric='minisblack')



#%% MERGE ALL ORIGINAL TIFFS STACKS TOGETHER INTO BIGGER TIFF FILE(S) FOR ANALYSIS

num_tiffs = 5  # number of tiff files from overall list for one recording to merge

# specify save path to write merged tiff to:
save_as = '/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/pre00001/2020-07-16_HF098_pre.tif'
with tf.TiffWriter(save_as, bigtiff=True) as tif:
    for i in range(0, num_tiffs):
        # specify location of tiffs to be merged (note the magic characters used to cycle through each tiff and read tiff
        tiff_path = '/mnt/qnap_Root/apacker/Data/2020-07-17/WF/HF098/2020-07-16_HF098_pre00001/2020-07-16_HF098_pre00001(%s).tif' % (i + 1)
        stack = tf.imread(tiff_path)
        for j in range(len(stack)):
            tif.save(stack[j])
        msg = ' -- Writing tiff: ' + str(i + 1) + ' out of ' + str(num_tiffs)
        print(msg, end='\r')

del(stack)


#%% MERGING INDIVIDUAL TIFFS FROM HF95

tiff_paths = '/mnt/qnap_Root/apacker/Data/2020-07-16/WF/wide4X4ap_010*.tif'
ao.make_tiff_stack(tiff_paths, save_as='/home/pshah/mnt/qnap/Analysis/2020-07-16/HF95/wide4x4ap_010.tif')

#%% IMPORT back in the merged tiff and measure avg signal across the whole field

tiff_path = "/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/2020-07-17_HF098_003_first33pct.tif"
stack_ = tf.imread(tiff_path)

ca_avg_ = np.mean(stack_, axis=2)
ca_avg = np.mean(ca_avg_, axis=1)

plt.figure(figsize=[10,2])
plt.plot(ca_avg, linewidth=0.2, c='green')
plt.show()

#%% starting to sync up LFP + Ca signal by matching up indexes from the paq file to imaging frames

## get start index of camera clock from the .paq file
# use threshold and then find large spaces in the difference between the indexes

frames = pjf.threshold_detect(camera, 1.0)

# the start and stops of camera frames are where the second derivative of frames exceeds the regular values
diff = np.diff(frames)

cam_stops = frames[np.where(diff > 1000)[0]]
cam_starts = frames[np.where(diff > 1000)[0]+1]

print('Total frames:', len(frames))
print("First camera clock index:", frames[0])
print("camera stop indexes:", cam_stops)
print("camera start indexes: ", cam_starts)
print("Last camera clock index:", frames[-1])

#%% specify a certain timepoint to identify the camera frame index and .paq sample indexes for
timepoint = 1.0e7  # this is a timepoint determined from the .paq voltage trace

# find the closest value in a list to the given input
sample_index, frame_index = pjf.findClosest(frames, timepoint)
print(sample_index)  # to select voltage recording index from paq file
print('frame of interest:', frame_index)  # find the corresponding frame # (remember that for this file the ca measurement was already taken from fr no.8404 onwards)

# subset the paq (voltage recording) and the calcium signal appropriately
volt_ = voltage[:int(sample_index)]
ca_avg_ = ca_avg[:frame_index]

# plot voltage and calcium trace - the two signals should be temporally synced
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6)); fig.subplots_adjust(hspace=0.5)
ax1.plot(volt_, linewidth=0.2)
ax1.set_ylabel('Voltage (mV?)')
ax2.plot(ca_avg_, linewidth=0.2, c='green')
ax2.set_ylabel('Ca2+ fluorescence')
plt.show()



#%% pre-4ap state; do you see UP and DOWN states in LFP and Ca imaging?



#%% measuring ROIs from Ca imaging data
tiff_path = "/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/2020-07-17_HF098_003_first33pct.tif"
stack = tf.imread(tiff_path); stack = stack[16000:]
# stack = stack_

x_pos = 746
y_pos = 1060
width = 200
height = width

stack_roi_1 = stack[:, 623:623+width, 577:577+height]
stack_roi_2 = stack[:, 636:636+width, 384:384+height]
stack_roi_3 = stack[:, 400:400+width, 393:393+height]
stack_roi_4 = stack[:, 378:378+width, 554:554+height]

stack_roi_1.shape


# SAVING ROI TRACES AS PICKLE FILES - CAN SKIP OUT ONCE COMPLETED
pkl.dump(stack_roi_1, open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_1-2020-07-17_HF098_003_first33pct.pkl', 'wb'))
pkl.dump(stack_roi_2, open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_2-2020-07-17_HF098_003_first33pct.pkl', 'wb'))
pkl.dump(stack_roi_3, open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_3-2020-07-17_HF098_003_first33pc.pkl', 'wb'))
pkl.dump(stack_roi_4, open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_4-2020-07-17_HF098_003_first33pct.pkl', 'wb'))


# IMPORTING ROI TRACES FROM PICKLE FILES - MAKE SURE TO POINT PATH TO APPROPRIATE DIRECTORY
stack_roi_1 = pkl.load(open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_1-2020-07-17_HF098_003_first33pct.pkl', 'rb'))
stack_roi_2 = pkl.load(open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_2-2020-07-17_HF098_003_first33pct.pkl', 'rb'))
stack_roi_3 = pkl.load(open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_3-2020-07-17_HF098_003_first33pct.pkl', 'rb'))
stack_roi_4 = pkl.load(open('/home/pshah/mnt/qnap/Analysis/2020-07-17/HF98/003/roi_4-2020-07-17_HF098_003_first33pct.pkl', 'rb'))

#%%

roi_1 = stack_roi_1
roi_2 = stack_roi_4
title = 'ROI 1 and 4'

plt.figure(figsize=[10, 2])
ca_avg_ = np.mean(roi_1, axis=2)
ca_avg = np.mean(ca_avg_, axis=1)

ca_avg_2 = np.mean(roi_2, axis=2)
ca_avg2 = np.mean(ca_avg_2, axis=1)



yhat = savgol_filter(ca_avg, 51, 5)

plt.plot(ca_avg, linewidth=0.2, c='green')
plt.plot(ca_avg2, linewidth=0.2, c='blue')
plt.suptitle(title)

# ca_avg_ = np.mean(stack, axis=2)
# ca_avg = np.mean(ca_avg_, axis=1)
# plt.plot(ca_avg, linewidth=0.2, c='green')
plt.grid(True, which='both', axis='both')
plt.show()


#%% isolate seizure events based on selecting event #s from the LFP data, then we can isolate the Calcium ROI signal and then do cross-correlogram analysis

# cross-correlogram of the whole signal to test out this code

corr = signal.correlate(stack_roi_3, stack_roi_2, mode='full')

# # matplotlib example
# x, y = np.random.randn(2, 100)
#
# fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
# ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
# ax1.grid(True)
#
# ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
# ax2.grid(True)
#
# plt.show()

def genSine(f0, fs, phase, dur):
    t = np.arange(dur)
    sinusoid = np.sin(2*np.pi*t*(f0/fs) - phase) + np.random.rand(len(t))
    return sinusoid

x = genSine(10,1000, 0, 500)
x2 = genSine(10,1000, 100, 500)
plt.plot(x);plt.show()


plt.acorr(x, usevlines=True, normed=True, maxlags=100, lw=1)
plt.grid(True)
plt.show()

plt.xcorr(x, x2, usevlines=True, normed=True, maxlags=100, lw=1)
plt.grid(True)
plt.show()



# trying this approach with the calcium imaging data
plt.acorr(ca_avg, usevlines=True, normed=True, maxlags=50, lw=1)
plt.grid(True)
plt.show()



#%%
width_plot=100

plt.figure()
a1 = savgol_filter(np.mean(np.mean(stack_roi_2, axis=2), axis=1)[:], 51, 4)
a2 = savgol_filter(np.mean(np.mean(stack_roi_3, axis=2), axis=1)[:], 51, 4)
result = np.correlate(a1, a2, mode='full')
result_2 = result[(result.size + 1) // 2 - width_plot : (result.size + 1) // 2 + width_plot]

x_range_1 = np.arange(len(result_2)) - len(result_2)/2 + 1
plt.plot(x_range_1, result_2, color='blue')

plt.show()

#%%
pearsonr(np.mean(np.mean(stack_roi_2, axis=2), axis=1),
         np.mean(np.mean(stack_roi_3, axis=2), axis=1))

#%% autocorrelation function from the statsmodels library
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(x)
plt.show()
