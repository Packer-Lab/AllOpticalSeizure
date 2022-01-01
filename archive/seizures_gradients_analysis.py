## AN APPROACH TO LOOK AT THE DOMINANT DIRECTION OF TRAVEL OF THE SEIZURE IN CALCIUM IMAGING MOVIES

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc


#%%

shape = (512, 512)
angle = 40


# translate the image matrix largely into upper right quadrant so you have plenty of space to work on it

# play data
data = np.tile(0, [shape[0], shape[1]])
data[:256, :] = 100


#%%

fig = plt.figure(figsize=(10, 3))
ax1, ax2, ax3 = fig.subplots(1, 3)
# img = misc.ascent()
img = data
img_45 = ndimage.rotate(img, 45, reshape=False)
full_img_45 = ndimage.rotate(img, 45, reshape=True)
ax1.imshow(img, cmap='gray')
ax1.set_axis_off()
ax2.imshow(img_45, cmap='gray')
ax2.set_axis_off()
ax3.imshow(full_img_45, cmap='gray')
ax3.set_axis_off()
fig.set_tight_layout(True)
plt.show()

#%%
# calculating average of the image along the x axis

img = misc.ascent()


for i in [0, 45, 90, 135, 180]:
    full_img_rot = ndimage.rotate(img, i, reshape=True)

    avg = np.zeros([full_img_rot.shape[0], 1])
    for i in range(len(full_img_rot)):
        x = full_img_rot[i][full_img_rot[i]!=0]
        if x:
            avg[i] = x.mean()
        else:
            avg[i] = 0

    plt.plot(avg[20:-20])
plt.show()


#%% seizure drop

# plot derivative of the seizure wavefront image

