### this file contains a bunch of the code that is archived to keep for reference but not fully deleted



# %% ################ archive dump on Mar 17 2021

# THE FOLLOWING FUNCS HAVE BEEN MOVED TO all_optical_utils.py AS THEY ARE NOW MOSTLY STABLE
@njit
def moving_average(a, n=4):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@njit
def _good_cells(cell_ids, raws, photostim_frames, radiuses, min_radius_pix, max_radius_pix):
    good_cells = []
    len_cell_ids = len(cell_ids)
    for i in range(len_cell_ids):
        # print(i, " out of ", len(cell_ids), " cells")
        raw = raws[i]
        raw_ = np.delete(raw, photostim_frames)
        raw_dff = aoutils.normalize_dff_jit(raw_)
        std_ = raw_dff.std()

        raw_dff_ = moving_average(raw_dff, n=4)

        thr = np.mean(raw_dff) + 2.5 * std_
        e = np.where(raw_dff_ > thr)
        # y = raw_dff_[e]

        radius = radiuses[i]

        if len(e) > 0 and radius > min_radius_pix and radius < max_radius_pix:
            good_cells.append(cell_ids[i])

        if i % 100 == 0:  # print the progress once every 100k iterations
            print(i, " out of ", len_cell_ids, " cells done")
    print('# of good cells found: ', len(good_cells), ' (out of ', len_cell_ids, ' ROIs)')
    return good_cells


def _good_photostim_cells(expobj, groups_per_stim=1, std_thresh=1, dff_threshold=None, pre_stim=expobj.pre_stim,
                          post_stim=expobj.post_stim):
    '''
    make sure to specify std threshold to use for filtering
    the pre-stim and post-stim args specify which pre-stim and post-stim frames to consider for filtering
    '''
    expobj.good_photostim_cells = []
    expobj.good_photostim_cells_responses = []
    expobj.good_photostim_cells_stim_responses_dF_stdF = []
    expobj.good_photostim_cells_stim_responses_dFF = []
    total = 0  # use to tally up how many cells across all groups are filtered in
    total_considered = 0  # use to tally up how many cells were looked at for their photostim response.
    for group in range(len(expobj.s2p_photostim_targets)):
        print('\nGroup %s' % group)
        stim_timings = expobj.stim_start_frames[group::int(expobj.n_groups / groups_per_stim)]
        title = 'SLM photostim Group #%s' % group
        targeted_cells = [cell for cell in expobj.s2p_photostim_targets[group] if cell in expobj.good_cells]

        # collect photostim timed average dff traces of photostim targets
        targets_dff = []
        pre_stim = pre_stim
        post_stim = post_stim
        for cell in targeted_cells:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)
                flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                       stim not in expobj.seizure_frames]

                flu_dff = []
                for trace in flu:
                    mean = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean) / mean) * 100
                    flu_dff.append(trace_dff)

                targets_dff.append(np.mean(flu_dff, axis=0))

        # FILTER CELLS WHERE PHOTOSTIMULATED TARGETS FIRE > 1*std ABOVE PRE-STIM
        good_photostim_responses = {}
        good_photostim_cells = []
        good_targets_dF_stdF = []
        good_targets_dff = []
        std_thresh = std_thresh
        for cell in targeted_cells:
            trace = targets_dff[
                targeted_cells.index(cell)]  # trace = averaged dff trace across all photostims. for this cell
            pre_stim_trace = trace[:pre_stim]
            # post_stim_trace = trace[pre_stim + expobj.duration_frames:post_stim]
            mean_pre = np.mean(pre_stim_trace)
            std_pre = np.std(pre_stim_trace)
            # mean_post = np.mean(post_stim_trace[:10])
            dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
            # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
            response = np.mean(trace[
                               pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window
            if dff_threshold is None:
                thresh_ = mean_pre + std_thresh * std_pre
            else:
                thresh_ = mean_pre + dff_threshold  # need to triple check before using
            if response > thresh_:  # test if the response passes threshold
                good_photostim_responses[cell] = response
                good_photostim_cells.append(cell)
                good_targets_dF_stdF.append(dF_stdF)
                good_targets_dff.append(trace)
                print('Cell #%s - dFF post-stim: %s (threshold value = %s)' % (cell, response, thresh_))

        expobj.good_photostim_cells.append(good_photostim_cells)
        expobj.good_photostim_cells_responses.append(good_photostim_responses)
        expobj.good_photostim_cells_stim_responses_dF_stdF.append(good_targets_dF_stdF)
        expobj.good_photostim_cells_stim_responses_dFF.append(good_targets_dff)

        print('%s cells filtered out of %s s2p target cells' % (len(good_photostim_cells), len(targeted_cells)))
        total += len(good_photostim_cells)
        total_considered += len(targeted_cells)

    expobj.good_photostim_cells_all = [y for x in expobj.good_photostim_cells for y in x]
    print('\nTotal number of good photostim responsive cells found: %s (out of %s)' % (total, total_considered))


# box plot with overlaid scatter plot with seaborn
plt.rcParams['figure.figsize'] = (20, 7)
sns.set_theme(style="ticks")
sns.catplot(x='group', y='Avg. dF/stdF response', data=expobj.average_responses_dfstdf, alpha=0.8, aspect=0.75,
            height=3.5)
ax = sns.boxplot(x='group', y='Avg. dF/stdF response', data=expobj.average_responses_dfstdf, color='white', fliersize=0,
                 width=0.5)
for i, box in enumerate(ax.artists):
    box.set_alpha(0.3)
    box.set_edgecolor('black')
    box.set_facecolor('white')
    for j in range(6 * i, 6 * (i + 1)):
        ax.lines[j].set_color('black')
        ax.lines[j].set_alpha(0.3)
# plt.savefig("/Users/prajayshah/OneDrive - University of Toronto/UTPhD/Proposal/2020/Figures/target_responses_avg %s.svg" % experiment)
plt.setp(ax.get_xticklabels(), rotation=45)
plt.suptitle('%s' % experiment, fontsize=10)
plt.show()


def plot_flu_trace(expobj, idx, slm_group=None, to_plot='raw'):
    raw = expobj.raw[idx]
    raw_ = np.delete(raw, expobj.photostim_frames)
    raw_dff = aoutils.normalize_dff(raw_)
    std_dff = np.std(raw_dff, axis=0)
    std = np.std(raw_, axis=0)

    x = []
    # y = []
    for j in np.arange(len(raw_dff), step=4):
        avg = np.mean(raw_dff[j:j + 4])
        if avg > np.mean(raw_dff) + 2.5 * std_dff:
            x.append(j)
            # y.append(0)

    if to_plot == 'raw':
        to_plot_ = raw
        to_thresh = std
    elif to_plot == 'dff':
        to_plot_ = raw_dff
        to_thresh = std_dff
    else:
        ValueError('specify to_plot as either "raw" or "dff"')

    plt.figure(figsize=(20, 3))
    plt.plot(to_plot_, linewidth=0.1)
    if to_plot == 'raw':
        plt.suptitle(('raw flu for cell #%s' % expobj.cell_id[idx]), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=1.00)
    elif to_plot == 'dff':
        plt.scatter(x, y=[0] * len(x), c='r', linewidth=0.10)
        plt.axhline(y=np.mean(to_plot_) + 2.5 * to_thresh, c='green')
        plt.suptitle(('%s flu for cell #%s' % (to_plot, expobj.cell_id[idx])), horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=15, y=1.00)

    if slm_group is not None:
        for i in expobj.stim_start_frames[slm_group::expobj.n_groups]:
            plt.axvline(x=i - 1, c='gray', alpha=0.1)

    if len(expobj.seizure_frames) > 0:
        plt.scatter(expobj.seizure_frames, y=[-20] * len(x), c='g', linewidth=0.10)

    # plt.ylim(0, 300)
    plt.show()


def plot_photostim_avg(dff_array, stim_duration, pre_stim=10, post_stim=200, title='', y_min=None, y_max=None,
                       x_label=None, y_label=None, savepath=None):
    dff_array = dff_array[:, :pre_stim + post_stim]
    len_ = len(dff_array)
    flu_avg = np.mean(dff_array, axis=0)
    x = list(range(-pre_stim, post_stim))

    fig, ax = plt.subplots()
    ax.margins(0)
    ax.axvspan(0, stim_duration, alpha=0.2, color='green')
    for cell_trace in dff_array:
        ax.plot(x, cell_trace, linewidth=1, alpha=0.8)
    ax.plot(x, flu_avg, color='black', linewidth=2)  # plot median trace
    if y_min != None:
        ax.set_ylim([y_min, y_max])
    ax.set_title((title + ' - %s' % len_ + ' cells'), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # change x axis ticks to seconds
    labels = [item for item in ax.get_xticks()]
    for item in labels:
        labels[labels.index(item)] = round(item / expobj.fps, 1)
    ax.set_xticklabels(labels)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    plt.show()


def calculate_reliability(expobj, groups_per_stim=1, dff_threshold=20, pre_stim=expobj.pre_stim,
                          post_stim=expobj.post_stim):
    '''calculates the percentage of successful photoresponsive trials for each targeted cell, where success is post
     stim response over the dff_threshold'''
    reliability = {}  # dict will be used to store the reliability results for each targeted cell
    targets_dff_all_stimtrials = {}  # dict will contain the peri-stim dFF for each cell by the cell_idx
    stim_timings = expobj.stim_start_frames
    for group in range(len(expobj.s2p_photostim_targets)):
        print('\nProcessing Group %s' % group)
        if groups_per_stim == 1:
            stim_timings = expobj.stim_start_frames[group::expobj.n_groups]

        targeted_cells = [cell for cell in expobj.s2p_photostim_targets[group] if cell in expobj.good_cells]

        # collect photostim timed average dff traces of photostim targets
        for cell in targeted_cells:
            # print('considering cell # %s' % cell)
            if cell in expobj.cell_id:
                cell_idx = expobj.cell_id.index(cell)
                flu = [expobj.raw[cell_idx][stim - pre_stim: stim + post_stim] for stim in stim_timings if
                       stim not in expobj.seizure_frames]

                flu_dff = []
                success = 0
                for trace in flu:
                    # calculate dFF (noramlized to pre-stim) for each trace
                    mean = np.mean(trace[0:pre_stim])
                    trace_dff = ((trace - mean) / mean) * 100
                    flu_dff.append(trace_dff)

                    # calculate if the current trace beats dff_threshold for calculating reliability (note that this happens over a specific window just after the photostim)
                    response = np.mean(trace[
                                       pre_stim + expobj.duration_frames:pre_stim + 3 * expobj.duration_frames])  # calculate the dF over pre-stim mean F response within the response window
                    if response >= round(dff_threshold):
                        success += 1

                targets_dff_all_stimtrials[cell_idx] = np.array(
                    flu_dff)  # add the trials x peri-stim dFF as an array for each cell
                reliability[cell_idx] = success / len(stim_timings) * 100.
    print(reliability)
    return reliability, targets_dff_all_stimtrials


d = {}
# d['group'] = [int(expobj.good_photostim_cells.index(x)) for x in expobj.good_photostim_cells for y in x]
d['group'] = ['non-target'] * (len(expobj.good_cells))
for stim in expobj.stim_start_frames:
    d['%s' % stim] = [None] * len(expobj.good_cells)
df = pd.DataFrame(d, index=expobj.good_cells)
# population dataframe
for group in cell_groups:
    # hard coded number of stim. groups as the 0 and 1 in the list of this for loop
    if group == 'non-target':
        for stim in expobj.stim_start_frames:
            cells = [i for i in expobj.good_cells if i not in expobj.good_photostim_cells_all]
            for cell in cells:
                cell_idx = expobj.cell_id.index(cell)
                trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                trace_dff = ((trace - mean_pre) / abs(mean_pre))  # * 100
                std_pre = np.std(trace[0:expobj.pre_stim])
                # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
                response = np.mean(trace_dff[
                                   expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
                df.at[cell, '%s' % stim] = round(response, 4)
    elif 'SLM Group' in group:
        cells = expobj.good_photostim_cells[int(group[-1])]
        for stim in expobj.stim_start_frames:
            for cell in cells:
                cell_idx = expobj.cell_id.index(cell)
                trace = expobj.raw[cell_idx][stim - expobj.pre_stim:stim + expobj.duration_frames + expobj.post_stim]
                mean_pre = np.mean(trace[0:expobj.pre_stim])
                trace_dff = ((trace - mean_pre) / abs(mean_pre)) * 100
                std_pre = np.std(trace[0:expobj.pre_stim])
                # response = np.mean(trace_dff[pre_stim + expobj.duration_frames:pre_stim + 3*expobj.duration_frames])
                dF_stdF = (trace - mean_pre) / std_pre  # make dF divided by std of pre-stim F trace
                # response = np.mean(dF_stdF[pre_stim + expobj.duration_frames:pre_stim + 1 + 2 * expobj.duration_frames])
                response = np.mean(trace_dff[
                                   expobj.pre_stim + expobj.duration_frames:expobj.pre_stim + 1 + 2 * expobj.duration_frames])
                df.at[cell, '%s' % stim] = round(response, 4)
                df.at[cell, 'group'] = group


# moved to the alloptical_plotting.py file
def bar_with_points(data, title='', x_tick_labels=[], points=True, bar=True, colors=['black'], ylims=None, xlims=None,
                    x_label=None, y_label=None, alpha=0.2, savepath=None):
    """
    general purpose function for plotting a bar graph of multiple categories with the individual datapoints shown
    as well. The latter is achieved by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph.

    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param alpha: transparency of the individual points when plotted in the scatter
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :return: matplotlib plot
    """

    w = 0.3  # bar width
    x = list(range(len(data)))
    y = data

    # plt.figure(figsize=(2, 10))
    fig, ax = plt.subplots(figsize=(2 * len(x), 3))
    if not bar:
        for i in x:
            ax.plot(np.linspace(x[i] - w / 2, x[i] + w / 2, 3), [np.mean(yi) for yi in y] * 3, color=colors[i])
        lw = 0,
        edgecolor = None
    else:
        edgecolor = 'black',
        lw = 1

    # plot bar graph, or if no bar (when lw = 0) then use it to plot the error bars
    ax.bar(x,
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y],  # error bars
           capsize=3,  # error bar cap width in points
           width=w,  # bar width
           linewidth=lw,  # width of the bar edges
           # tick_label=x_tick_labels,
           edgecolor=edgecolor,
           color=(0, 0, 0, 0),  # face color transparent
           )
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)

    if xlims:
        ax.set_xlim([xlims[0] - 1, xlims[1] + 1])
    elif len(x) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims = [-1, 1]
        ax.set_xlim(xlims)

    if points:
        for i in x:
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i], alpha=alpha)

    if ylims:
        ax.set_ylim(ylims)
    elif len(x) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    ax.set_title((title), horizontalalignment='center', verticalalignment='top', pad=20,
                 fontsize=10)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if savepath:
        plt.savefig(savepath)
    plt.show()

def xyloc_responses(expobj, to_plot='dfstdf', clim=[-10, +10], plot_target_coords=True, save_fig: str = None):
    """
    plot to show the response magnitude of each cell as the actual's filling in the cell's ROI pixels.

    :param expobj:
    :param to_plot:
    :param clim:
    :param plot_target_coords: bool, if True plot the actual X and Y coords of all photostim cell targets
    :param save_fig: where to save the save figure (optional)
    :return:
    """
    stim_timings = [str(i) for i in
                    expobj.stim_start_frames]  # need each stim start frame as a str type for pandas slicing

    if to_plot == 'dfstdf':
        average_responses = expobj.dfstdf_all_cells[stim_timings].mean(axis=1).tolist()
    elif to_plot == 'dff':
        average_responses = expobj.dff_all_cells[stim_timings].mean(axis=1).tolist()
    else:
        raise Exception('need to specify to_plot arg as either dfstdf or dff in string form!')

    # make a matrix containing pixel locations and responses at each of those pixels
    responses = np.zeros((expobj.frame_x, expobj.frame_x), dtype='uint16')

    for n in expobj.good_cells:
        idx = expobj.cell_id.index(n)
        ypix = expobj.stat[idx]['ypix']
        xpix = expobj.stat[idx]['xpix']
        responses[ypix, xpix] = 100. + 1 * round(average_responses[expobj.good_cells.index(n)], 2)

    # mask some 'bad' data, in your case you would have: data < 0.05
    responses = np.ma.masked_where(responses < 0.05, responses)
    cmap = plt.cm.bwr
    cmap.set_bad(color='black')

    plt.figure(figsize=(7, 7))
    im = plt.imshow(responses, cmap=cmap)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(to_plot)

    plt.clim(100+clim[0], 100+clim[1])
    if plot_target_coords:
        for (x, y) in expobj.target_coords_all:
            plt.scatter(x=x, y=y, edgecolors='green', facecolors='none', linewidths=1.0)
    plt.suptitle((experiment + ' - avg. dFF - targets in green'), y=0.95, fontsize=10)
    # pj.plot_cell_loc(expobj, cells=expobj.s2p_cell_targets, background_transparent=True)
    plt.show()
    if save_fig is not None:
        plt.savefig(save_fig)


# moved to the alloptical_processing_photostim.py file
# ###### IMPORT pkl file containing expobj

# determine which frames to retrieve from the overall total s2p output
trials = ['t-005', 't-006', 't-008', 't-009', 't-010']
total_frames_stitched = 0;
fr_curr_trial = None
for trial_ in trials:
    pkl_path_2 = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)
    with open(pkl_path_2, 'rb') as f:
        expobj = pickle.load(f)
        # import suite2p data
    total_frames_stitched += expobj.n_frames
    if trial_ == trial:
        fr_curr_trial = [total_frames_stitched - expobj.n_frames, total_frames_stitched]

with open(pkl_path, 'rb') as f:
    expobj = pickle.load(f)

# suite2p processing on expobj

s2p_path = '/home/pshah/mnt/qnap/Analysis/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# s2p_path = '/Users/prajayshah/Documents/data-to-process/2020-12-18/suite2p/alloptical-2p-pre-4ap-08x/plane0'
# flu, spks, stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)


# s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-18/suite2p/photostim-4ap_stitched/plane0'
expobj.s2pProcessing(s2p_path=s2p_path, subset_frames=fr_curr_trial, subtract_neuropil=True)
# if needed for pkl expobj generated from older versions of Vape
expobj.target_coords_all = expobj.target_coords
expobj.s2p_targets()

# expobj.target_coords_all = expobj.target_coords

# flu, expobj.spks, expobj.stat = uf.s2p_loader(s2p_path, subtract_neuropil=True)

aoutils.s2pMaskStack(obj=expobj, pkl_list=[pkl_path], s2p_path=s2p_path,
                     parent_folder='/home/pshah/mnt/qnap/Analysis/2020-12-18/')

# %% (quick) plot individual fluorescence traces - see InteractiveMatplotlibExample to make these plots interactively
# # plot raw fluorescence traces
# plt.figure(figsize=(50,3))
# for i in expobj.s2p_cell_targets:
#     plt.plot(expobj.raw[i], linewidth=0.1)
# plt.xlim(0, len(expobj.raw[0]))
# plt.show()

# plotting the distribution of radius and aspect ratios - should this be running before the filtering step which is right below????????

to_plot = plot_cell_radius_aspectr(expobj, expobj.stat, to_plot='radius')
a = [i for i in to_plot if i > 6]
id = to_plot.index(min(a))
# expobj.good_cells[id]

id = expobj.cell_id.index(1937)
expobj.stat[id]

# ###### CODE TO FILTER CELLS THAT ARE ACTIVE AT LEAST ONCE FOR >2.5*std

# pull out needed variables because numba doesn't work with custom classes (such as this all-optical class object)
cell_ids = expobj.cell_id
raws = expobj.raw
# expobj.append_seizure_frames(bad_frames=None)
photostim_frames = expobj.photostim_frames
radiuses = expobj.radius

# initial quick run to allow numba to compile the function - not sure if this is actually creating time savings
_ = aoutils._good_cells(cell_ids=cell_ids[:3], raws=raws, photostim_frames=expobj.photostim_frames, radiuses=radiuses,
                        min_radius_pix=2.5, max_radius_pix=8.5)
expobj.good_cells = aoutils._good_cells(cell_ids=cell_ids, raws=raws, photostim_frames=expobj.photostim_frames,
                                        radiuses=radiuses,
                                        min_radius_pix=2.5, max_radius_pix=8.5)

# filter for GOOD PHOTOSTIM. TARGETED CELLS with responses above threshold

expobj.pre_stim = 15  # specify pre-stim and post-stim periods of analysis and plotting
expobj.post_stim = 150

# function for gathering all good photostim cells who respond on average across all trials to the photostim
# note that the threshold for this is 1 * std of the prestim raw flu (fluorescence trace)

aoutils._good_photostim_cells(expobj=expobj, groups_per_stim=3, pre_stim=expobj.pre_stim, post_stim=expobj.post_stim,
                              dff_threshold=None)

# (full) plot individual cell's flu or dFF trace, with photostim. timings for that cell
cell = 1
# group = [expobj.good_photostim_cells.index(i) for i in expobj.good_photostim_cells if cell in i][0]  # this will determine which slm group's photostim to plot on the flu trace
group = 1

# plot flu trace of selected cell with the std threshold
idx = expobj.cell_id.index(cell)
plot_flu_trace(expobj=expobj, idx=idx, slm_group=group, to_plot='dff');
print(expobj.stat[idx])

#
##### SAVE expobj as PKL
# Pickle the expobject output to save it for analysis
pkl_path = "/home/pshah/mnt/qnap/Data/%s/%s_%s/%s_%s.pkl" % (date, date, trial, date, trial)


def save_pkl(expobj, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(expobj, f)
    print("pkl saved to %s" % pkl_path)


def find_photostim_frames(expobj):
    '''finds all photostim frames and saves them into the bad_frames.npy file'''
    photostim_frames = []
    for j in expobj.stim_start_frames:
        for i in range(
                expobj.duration_frames + 1):  # usually need to remove 1 more frame than the stim duration, as the stim isn't perfectly aligned with the start of the imaging frame
            photostim_frames.append(j + i)

    expobj.photostim_frames = photostim_frames
    # print(photostim_frames)
    print('/// Original # of frames:', expobj.n_frames, 'frames ///')
    print('/// # of Photostim frames:', len(photostim_frames), 'frames ///')
    print('/// Minus photostim. frames total:', expobj.n_frames - len(photostim_frames), 'frames ///')


find_photostim_frames(expobj)

###### CODE TO FILTER TRIALS WHERE PHOTOSTIMULATED CELLS FIRE > 1*std ABOVE PRE-STIM
good_photostim_trials = []
good_photostimtrials_dF_stdF = []
good_photostimtrials_dFF = []
photostimtrials_dF_stdF = []
photostimtrials_dFF = []
for i in stim_timings:
    photostimcells_dF_stdF = []
    photostimcells_dFF = []
    for cell_idx in targets:
        trace = expobj.raw[expobj.cell_id.index(cell_idx)][
                i - pre_stim:i + post_stim]  # !!! cannot use the raw trace <-- avg of from all photostimmed cells
        pre_stim_trace = trace[:pre_stim]
        post_stim_trace = trace[pre_stim + 10:post_stim]
        mean_pre = np.mean(pre_stim_trace)
        std_pre = np.std(pre_stim_trace)
        mean_post = np.mean(post_stim_trace)
        dF_stdF = (trace - mean_pre) / std_pre
        dFF = (trace - mean_pre) / mean_pre
        photostimcells_dF_stdF.append(dF_stdF)
        photostimcells_dFF.append(dFF)

    trial_dF_stdF = np.mean(photostimcells_dF_stdF, axis=0)
    trial_dFF = np.mean(photostimcells_dFF, axis=0)

    photostimtrials_dF_stdF.append(trial_dF_stdF)
    photostimtrials_dFF.append(trial_dFF)

    thresh = np.mean(trial_dFF[pre_stim + 10:pre_stim + 20])
    thresh_ = np.mean(trial_dF_stdF[pre_stim + 10:pre_stim + 20])

    # if thresh > 0.3:
    #     good_photostim_trials.append(i)

    if thresh_ > 1:
        good_photostim_trials.append(i)
        good_photostimtrials_dF_stdF.append(trial_dF_stdF)
        good_photostimtrials_dFF.append(trial_dFF)

# check to see what the new trials' photostim response look like
targets_dff_filtered = []
pre_stim = 10
post_stim = 250
for cell in targets:
    if cell in expobj.cell_id:
        cell_idx = expobj.cell_id.index(cell)
        flu = []
        for stim in good_photostim_trials:
            # frames_to_plot = list(range(stim-8, stim+35))
            flu.append(expobj.raw[cell_idx][stim - pre_stim:stim + post_stim])

        flu_dff = []
        for trace in flu:
            mean = np.mean(trace[0:pre_stim])
            trace_dff = ((trace - mean) / mean) * 100
            flu_dff.append(trace_dff)

        targets_dff_filtered.append(np.mean(flu_dff, axis=0))

        # flu_avg = np.mean(flu_dff, axis=0)
        # std = np.std(flu_dff, axis=0)
        # ci = 1.960 * (std/np.sqrt(len(flu_dff))) # 1.960 is z for 95% confidence interval, standard deviation divided by the sqrt of N samples (# traces in flu_dff)
        # x = list(range(-pre_stim, post_stim))
        # y = flu_avg
        #
        # fig, ax = plt.subplots()
        # ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1) # plot confidence interval
        # ax.axvspan(0, 10, alpha=0.2, color='red')
        # ax.plot(x, y)
        # fig.suptitle('Cell %s' % cell)
        # plt.show()

aoutils.plot_photostim_avg(dff_array=targets_dff_filtered, pre_stim=pre_stim, post_stim=post_stim, title=title)
aoutils.plot_photostim_(dff_array=targets_dff_filtered, pre_stim=pre_stim, post_stim=post_stim, title=title)

# now plot to see what the dF_stdF trace looks like
plot_photostim_(dff_array=good_photostimtrials_dFF, pre_stim=pre_stim, post_stim=post_stim, title='good trials dFF')
plot_photostim_(dff_array=photostimtrials_dFF, pre_stim=pre_stim, post_stim=post_stim, title='dFF')

fig, ax = plt.subplots()
for i in range(len(photostimtrials_dFF)):
    plt.plot(photostimtrials_dFF[i], linewidth=1.05)
ax.axvspan(10, 20, alpha=0.2, color='red')
plt.show()

fig, ax = plt.subplots()
plt.plot(trial_dFF)
ax.axvspan(10, 20, alpha=0.2, color='red')
plt.show()





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

