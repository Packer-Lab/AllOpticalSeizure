### this file contains a bunch of the code that is archived to keep for reference but not fully deleted

#%%
# deprecated for loop scan code below - above code with cumsum moving average should be much faster
# @njit
# def _good_cells(cell_ids, raws, photostim_frames, radiuses, min_radius_pix, max_radius_pix):
#     good_cells = []
#     for i in range(len(cell_ids)):
#         print(i, " out of ", len(cell_ids), " cells")
#         raw = raws[i]
#         raw_ = np.delete(raw, photostim_frames)
#         raw_dff = ao.normalize_dff_jit(raw_)
#         std_ = raw_dff.std()
#
#         a = []
#         y = []
#         for j in np.arange(len(raw_dff), step=4):  # think about whether or not you need to change this number to be dependent on the imaging fps
#             avg = np.mean(raw_dff[j:j+4])
#             if avg > np.mean(raw_dff)+2.5*std_: # if the avg of 4 frames is greater than the threshold then save the result
#                 a.append(j)
#                 y.append(avg)
#
#         radius = radiuses[i]
#
#         if len(a) > 0 and radius > min_radius_pix and radius < max_radius_pix:
#             good_cells.append(cell_ids[i])
#     print('# of good cells found: ', len(good_cells), ' (out of ', len(cell_ids), ' ROIs)')
#     return good_cells

#%%
# # use the code below for when you've stitched multiple trials together in suite2p
# pkl_path = '/Users/prajayshah/Documents/data-to-process/2020-03-18/2020-03-18_t-019.pkl'
# with open(pkl_path, 'rb') as f:
#     expobj_1 = pickle.load(f)
# pkl_path = '/Users/prajayshah/Documents/data-to-process/2020-03-18/2020-03-18_t-020.pkl'
# with open(pkl_path, 'rb') as f:
#     expobj_2 = pickle.load(f)
#
# # add photostim frames from expobj 2 to expobj 1 using extend
# new = []
# for i in expobj_2.photostim_frames:
#     new.append(i+expobj_1.n_frames)
# expobj_1.photostim_frames.extend(new)
#
# # make new expobj object with extended photostim frames
# expobj = expobj_1
#
# # add other bad imaging frames
# expobj.photostim_frames.extend(list(range(2680,5120))) # seizure/CSD frames

#%%

# make plots of photostim targeted trials
# def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
#                        x_label=None, y_label=None):
#     dff_array = dff_array[:,:pre_stim + post_stim]
#     len_ = len(dff_array)
#     flu_avg = np.median(dff_array, axis=0)
#     std = np.std(dff_array, axis=0)
#     ci = 1.960 * (std / np.sqrt(len_))  # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
#     x = list(range(-pre_stim, post_stim))
#     y = flu_avg
#
#     fig, ax = plt.subplots()
#     ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)  # plot confidence interval
#     ax.axvspan(0, stim_duration, alpha=0.2, color='green')
#     ax.plot(x, y)
#     if y_min != None:
#         ax.set_ylim([y_min, y_max])
#
#     ax.set_title((title+' - %s' % len_+' cells'), horizontalalignment='center', verticalalignment='top', pad=20, fontsize=10)
#
#     # change x axis ticks to seconds
#     labels = [item for item in ax.get_xticks()]
#     for item in labels:
#         labels[labels.index(item)] = int(round(item/expobj.fps))
#     ax.set_xticklabels(labels)
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     plt.show()
#
#
# plot_photostim_avg(dff_array=x, stim_duration=expobj.duration_frames, pre_stim=pre_stim, post_stim=post_stim,
#                    title=(experiment + '- %s avg. response of good responsive cells' % title), y_label=y_label, x_label='Time post-stimulation (seconds)')

#%%

