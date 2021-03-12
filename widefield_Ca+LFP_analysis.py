
sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/Vape')
sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_pj')
import utils.utils_funcs as uf #from Vape
import funcs_pj as pjf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats

# set up plotting settings to give real time value information
plt.rcParams['figure.figsize'] = [20, 3]

## 2020-03-06: t05 (LFP and 4AP on opposite hemispheres)

#%%
input_path = '/Volumes/Extreme SSD/oxford-data/2020-03-06/2020-03-06_HF82_t05.paq'
experiment = '4ap seizures - Widefield Ca + LFP'

paq = pjf.paq_read(input_path, plot=True)
camera_frames = paq['data'][1]
voltage = paq['data'][2]

## get start frames of camera clock
# use threshold and then find large spaces in the difference between the indexes

frames = pjf.threshold_detect(camera_frames, 1.0)

# the start and stops of camera frames are where the second derivative of frames exceeds the regular values
diff = np.diff(frames)

cam_stops = frames[np.where(diff > 1000)[0]]
cam_starts = frames[np.where(diff > 1000)[0]+1]

print('Total frames:', len(frames))
print("First camera clock index:", frames[0])
print("Camera stop indexes:", cam_stops)
print("Camera start indexes: ", cam_starts)
print("last camera clock index:", frames[-1])

# bring in Ca imaging rois now
t05_roi1 = np.loadtxt('/Volumes/Extreme SSD/oxford-data/2020-03-06/t05_roi1.txt', skiprows=1)[:,1]
t05_roi2 = np.loadtxt('/Volumes/Extreme SSD/oxford-data/2020-03-06/t05_roi2.txt', skiprows=1)[:,1]
t05_roi3 = np.loadtxt('/Volumes/Extreme SSD/oxford-data/2020-03-06/t05_roi3.txt', skiprows=1)[:,1]

# subset voltage and camera measurements to region of interest (imaging measurements taken from frame #8404 onwards)
frames = frames[8404:]

# find the closest value in a list to the given input
sample_index, frame_index = pjf.findClosest(frames, 9000000)

#%%
# subset appropriately
print(sample_index)  # to select voltages
print('frame of interest:', frame_index)  # find the corresponding frame # (remember that for this file the ca measurement was already taken from fr no.8404 onwards)

volt_ = voltage[sample_index:]  # TODO double check the subsetting here!!
t05_roi1_ = t05_roi1[(frame_index):]
t05_roi2_ = t05_roi2[(frame_index):]
t05_roi3_ = t05_roi3[(frame_index):]

#%%
plt.plot(volt_)
plt.show()

#%%

# normalize ca2+ signal
baseline = np.percentile(t05_roi3_, 10); print(baseline)
t05_roi3_dff = pjf.dff(t05_roi3_,baseline)

baseline = np.percentile(t05_roi2_, 10); print(baseline)
t05_roi2_dff = pjf.dff(t05_roi2_,baseline)

baseline = np.percentile(t05_roi1_, 10); print(baseline)
t05_roi1_dff = pjf.dff(t05_roi1_,baseline)



#%%
ca_imaging_1 = t05_roi3_dff
ca_imaging_2 = t05_roi2_dff
ca_imaging_3 = t05_roi1_dff
print('t05_roi3_, t05_roi2, t05_roi1:')
print('Frames: %d to %d' % (frame_index, frame_index+len(ca_imaging_1)))


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,12)); fig.subplots_adjust(hspace=0.5)
ax1.plot(volt_, color='red')
ax1.set_ylabel('Voltage (mV?)')
ax2.plot(ca_imaging_1)
ax2.set_ylabel('Ca2+ fluorescence (dF/F)')
ax3.plot(ca_imaging_2)
ax3.set_ylabel('Ca2+ fluorescence (dF/F)')
ax4.plot(ca_imaging_3)
ax4.set_ylabel('Ca2+ fluorescence (dF/F)')
plt.show()


#%%

# resample voltage data to same length as ca_imaging

ca_upsampled_1 = signal.resample(ca_imaging_1, len(volt_))
ca_upsampled_2 = signal.resample(ca_imaging_2, len(volt_))
ca_upsampled_3 = signal.resample(ca_imaging_3, len(volt_))


#%% DETRENDING LFP SIGNAL

# plot LFP signal with GCaMP signal
print('t05_roi3_, t05_roi2, t05_roi1:')
print('Frames: %d to %d' % (frame_index ,frame_index+len(ca_imaging_1)))

# define x range in time
x = np.linspace(0, len(volt_)/10000, len(volt_))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(16,8)); fig.subplots_adjust(hspace=0.2)
#ax1.plot(volt_)
ax1.plot(signal.detrend(volt_), color='red')
ax1.set_ylabel('Voltage (mV?)')
ax2.plot(x, ca_upsampled_1)
ax2.set_ylabel('Ca2+ fluorescence (dF/F)')
ax2.legend(['LFP electrode - GCaMP'], loc=1)
ax3.plot(x, ca_upsampled_2)
ax3.set_ylabel('Ca2+ fluorescence')
ax3.legend(['ipsilateral hemisphere'], loc=1)
ax4.plot(x, ca_upsampled_3)
ax4.set_ylabel('Ca2+ fluorescence')
ax4.legend(['contralateral hemisphere'], loc=1)
ax4.set_xlabel('Time (s)')
plt.suptitle(experiment, y=1.00)
for ax in (ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# plt.savefig('/Volumes/Extreme SSD/oxford-data/2020-03-06/Ca_LFP_2020-03-06.svg', format='svg')
for ax in (ax2, ax3, ax4):
    ax.set_ylim((-0.5, 1.8))
plt.savefig("/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/%s.svg" % experiment)
plt.show()

#%%

# calculate pearson correlation of LFP signal and ca_imaging signal

r_lfp_roi3 = stats.pearsonr(volt_, ca_upsampled_1); print('r| LFP vs. roi3:  %f' % r_lfp_roi3[0])
r_lfp_roi2 = stats.pearsonr(volt_, ca_upsampled_2); print('r| LFP vs. roi2:  %f' % r_lfp_roi2[0])
r_lfp_roi1 = stats.pearsonr(volt_, ca_upsampled_3); print('r| LFP vs. roi1:  %f' % r_lfp_roi1[0])
r_roi2_roi3 = stats.pearsonr(ca_upsampled_2, ca_upsampled_1); print('r| roi2 vs. roi3:  %f' % r_roi2_roi3[0])

#%%

r_lfp_roi1
