## PLOTTING OF LINES ROTATED AT VARIOUS ANGLES AND USED FOR SUBSELECTING DATA MATRICES -- GAVE UP ON AT SOME POINT
## FOR THE SEIZURES IMAGING FOV ANALYSIS


#%%
import numpy as np
import matplotlib.pyplot as plt


def find_lines_thru(shape, angle, intervals):

    """this function will be used to retrieve the indexes for a straight line through the specified shape at the
    given angle. note: the line will always go through the center of the shape
    shape: a 2d tuple of the length of each side of the shape
    angle: the angle through which to
    """

    angle = np.deg2rad(angle)
    # # initialize variables for testing
    # # use shape of (512, 512) for testing
    # shape = (512, 512)
    # angle = np.deg2rad(-30)
    # intervals = 30


    # THE KEY to getting mulitple parallel lines is to rotate the center by the same transformation angle and then translate
    # result_mtx along multiple centers along one axis

    # define center of shape
    center = (shape[0] / 2, shape[1] / 2)

    b = max(shape)  # length of the longest side of shape
    input_mtx = np.array([
        [0] * int(b/2. * 2**(1 / 2) + 1),  # the  + 1 is just there to ensure that the length is at least b/2. * 2**(1 / 2) after convertin to int which will round the output of (b/2. * 2**(1 / 2))
        range(0, int(b/2. * 2**(1 / 2) + 1))
    ])  # array of dim 2 x b*sqrt(2)/2

    rotation_mtx = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    result_mtx = np.matmul(rotation_mtx, input_mtx).astype(int)

    # a_ = np.array([-1 * input_mtx[0], -1 * input_mtx[1]])
    # input_mtx_ = np.hstack((input_mtx, a_))


    d_ = -1 * result_mtx
    result_mtx = np.hstack((result_mtx, d_))

    # move the result_mtx line to the center of the shape via translation

    result_mtx_cent = np.array([result_mtx[0] + center[0], result_mtx[1] + center[1]])


    ### make mulitple lines that are parallel of result_mtx
    # first rotate result_mtx_ pi/2
    ortho_rotation_mtx = np.array([
        [np.cos(np.pi/2), -np.sin(np.pi/2)],
        [np.sin(np.pi/2), np.cos(np.pi/2)]
    ])
    ortho_result_mtx = np.matmul(ortho_rotation_mtx, result_mtx).astype(int)

    ortho_result_mtx_cent = np.array([ortho_result_mtx[0] + center[0], ortho_result_mtx[1] + center[1]])


    # select n spots along ortho_result_mtx to use as the new center for result_mtx_
    coords_ = ortho_result_mtx_cent[:, ::intervals]

    # carry out the translation of result_mtx_ across all coords, and save results in a new matrix of length # of intervals
    result_mtx_translated = np.empty([coords_.shape[1], 2, int(b * 2**(1 / 2) + 2)])
    for i in range(coords_.shape[1]):
        result_mtx_translated[i] = np.array([
            result_mtx[0] + coords_[0, i],
            result_mtx[1] + coords_[1, i]
        ])

    return result_mtx_cent, result_mtx_translated.astype(int), ortho_result_mtx_cent, coords_

def select_data_lines(data_mtx, selection_mtx):
    """this function will take the data mtx and return only the data points the """

#%%

# constants
shape = (512, 512)
ANGLE = 45
INTERVALS = 30


center = (shape[0] / 2, shape[1] / 2)
b = max(shape)  # length of the longest side of shape

input_mtx = np.array([
    [0] * int(b / 2. * 2 ** (1 / 2) + 1),
    # the  + 1 is just there to ensure that the length is at least b/2. * 2**(1 / 2) after convertin to int which will round the output of (b/2. * 2**(1 / 2))
    range(0, int(b / 2. * 2 ** (1 / 2) + 1))
])  # array of dim 2 x b*sqrt(2)/2

a_ = np.array([-1 * input_mtx[0], -1 * input_mtx[1]])
input_mtx_ = np.hstack((input_mtx, a_))

result_mtx_cent, result_mtx_translated, ortho_result_mtx_cent, coords_ = find_lines_thru(shape=shape,
                                                                                         angle=ANGLE,
                                                                                         intervals=INTERVALS)

plt.figure(figsize=(10, 10))
plt.scatter(x=input_mtx_[0] + center[0], y=input_mtx_[1] + center[1], c='blue')
plt.scatter(x=result_mtx_cent[0], y=result_mtx_cent[1] , c='green')
plt.scatter(x=ortho_result_mtx_cent[0], y=ortho_result_mtx_cent[1] , c='pink')
# plt.scatter(x=coords_[0], y=coords_[1] , c='red')
plt.vlines([center[0] - shape[0]/2, center[0] + shape[0]/2], 0, shape[0], colors='r')
plt.hlines([center[0] - shape[1]/2, center[0] + shape[1]/2], 0, shape[1], colors='r')
# plt.show()

for i in range(coords_.shape[1]):
    print(np.deg2rad(ANGLE))
    plt.scatter(x=result_mtx_translated[i][0], y=result_mtx_translated[i][1] , c='black',
                s=0.5)
plt.show()


#%% selecting data from the imaging FOV using result_mtx_translated

# play data
data = np.tile(100, [512, 512])

zero_array = np.zeros([int(b * 2 ** (1 / 2) + 1), int(b * 2 ** (1 / 2) + 1)])
zero_array = np.zeros([800, 800])
center_ = (int(zero_array.shape[0] / 2), int(zero_array.shape[1] / 2))
# zero_array[center_[0]-512//2 : center_[0]+512//2, center_[1]-512//2 : center_[1]+512//2] = 1


for i in range(coords_.shape[1]):
    for x, y in zip(result_mtx_translated[i][0], result_mtx_translated[i][1]):
        # if center_[0]-512//2 < x < center_[0]+512//2 and center_[1]-512//2 < y < center_[1]+512//2:
        zero_array[y, x] = 1
selection_mtx = zero_array[:512, :512]  # the selected coords that you want have a value of 1 in this array

# plotting the new matrix with imshow
plt.figure(figsize=[10, 10])
plt.imshow(selection_mtx, cmap='gray')
# plt.xlim([400-512//2,400+512//2])
# plt.ylim([400-512//2,400+512//2])
plt.show()

#%% multiply the selection_mtx with the data array

data_sub = np.multiply(data, selection_mtx)


plt.figure(figsize=[10, 10])
plt.imshow(data, cmap='gray')
# plt.xlim([400-512//2,400+512//2])
# plt.ylim([400-512//2,400+512//2])
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(data_sub, cmap='gray')
# plt.xlim([400-512//2,400+512//2])
# plt.ylim([400-512//2,400+512//2])
plt.show()


#%%
def seizure_drop(self, image, indexes):
    """use the indexes to collect data from an image. use this data to calculate the precipitation of the
    fluroscence values across one axis of the indexes"""
    pass