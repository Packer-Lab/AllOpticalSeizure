# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import numpy as np
import pandas as pd
from scipy import stats, signal
import statsmodels.api
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)




# %% 5.5.2) # # PLOT - average response stim graph for positive and negative followers
# - make one graph per comparison for now... then can figure out how to average things out later.


experiments = ['RL108t', 'RL109t', 'PS05t', 'PS07t', 'PS06t', 'PS11t']
pre4ap_pos = []
pre4ap_neg = []
post4ap_pos = []
post4ap_neg = []

# positive responders
ncols = 3
nrows = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:

            mean_pre4ap_ = allopticalResults.possig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.possig_responders_traces[row][1]

            # plot avg with confidence intervals
            # fig, ax = plt.subplots()

            ax = axs[counter//ncols, counter % ncols]

            meanst = np.mean(mean_pre4ap_, axis=0)
            std = np.std(mean_pre4ap_, axis=0, ddof=1)
            ## change xaxis to time (secs)
            if len(meanst) < 100:
                fps = 15
            else:
                fps = 30
            x_time = np.linspace(0, len(meanst) / fps, len(meanst)) - allopticalResults.pre_stim_sec  # x scale, but in time domain (transformed from frames based on the provided fps)

            ax.plot(x_time, meanst, color='black', lw=2)
            ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.15, color='gray')

            meanst = np.mean(mean_post4ap_, axis=0)
            std = np.std(mean_post4ap_, axis=0, ddof=1)
            ## change xaxis to time (secs)
            if len(meanst) < 100:
                fps = 15
            else:
                fps = 30
            x_time = np.linspace(0, len(meanst) / fps, len(meanst)) - allopticalResults.pre_stim_sec  # x scale, but in time domain (transformed from frames based on the provided fps)

            ax.plot(x_time, meanst, color='green', lw=2)
            ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.15, color='green')

            ax.legend([f"pre4ap {mean_pre4ap_.shape[0]} cells", f"post4ap {mean_post4ap_.shape[0]} cells"])
            ax.set_ylim([-0.5, 1.0])
            ax.set_ylabel('dF/pre_stdF')
            ax.set_title(f"{allopticalResults.num_sig_responders_df.index[row]}")

            counter += 1
fig.suptitle('Avg. positive responders')
fig.show()


# negative responders
ncols = 3
nrows = 4
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 14))
counter = 0
for exp in experiments:
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:

            mean_pre4ap_ = allopticalResults.negsig_responders_traces[row][0]
            mean_post4ap_ = allopticalResults.negsig_responders_traces[row][1]

            # plot avg with confidence intervals
            # fig, ax = plt.subplots()

            ax = axs[counter//ncols, counter % ncols]


            meanst = np.mean(mean_pre4ap_, axis=0)
            std = np.std(mean_pre4ap_, axis=0, ddof=1)
            ## change xaxis to time (secs)
            if len(meanst) < 100:
                fps = 15
            else:
                fps = 30
            x_time = np.linspace(0, len(meanst) / fps, len(meanst)) - allopticalResults.pre_stim_sec  # x scale, but in time domain (transformed from frames based on the provided fps)

            ax.plot(x_time, meanst, color='black', lw=2)
            ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.15, color='gray')

            meanst = np.mean(mean_post4ap_, axis=0)
            std = np.std(mean_post4ap_, axis=0, ddof=1)
            ## change xaxis to time (secs)
            if len(meanst) < 100:
                fps = 15
            else:
                fps = 30
            x_time = np.linspace(0, len(meanst) / fps, len(meanst)) - allopticalResults.pre_stim_sec  # x scale, but in time domain (transformed from frames based on the provided fps)

            ax.plot(x_time, meanst, color='red', lw=2)
            ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.15, color='red')

            ax.legend(['pre4ap', 'post4ap'])
            ax.set_ylim([-0.5, 0.2])
            ax.set_ylabel('dF/pre_stdF')
            ax.set_title(f"{allopticalResults.num_sig_responders_df.index[row]}")


            counter += 1
fig.suptitle('Avg. negative responders')
fig.show()

    ### below is for trying to average between trials from the same experiments... not fully working yet...
    # for exp in experiments:
    #     rows = []
    #     for row in range(len(allopticalResults.num_sig_responders_df.index)):
    #         if exp in allopticalResults.num_sig_responders_df.index[row]:
    #             rows.append(row)
    # mean_pre4ap_ = [allopticalResults.possig_responders_traces[row][0] for row in rows]
    # mean_post4ap_ = [allopticalResults.possig_responders_traces[row][1] for row in rows]
    # for i in range(len(mean_pre4ap_) - 1):
    #     mean_pre4ap = np.append(mean_pre4ap_[0], mean_pre4ap_[i + 1], axis=0)
    #
    # for i in range(len(mean_post4ap_) - 1):
    #     mean_post4ap = np.append(mean_post4ap_[0], mean_post4ap_[i + 1], axis=0)

    # # plot avg with confidence intervals
    # fig, ax = plt.subplots()
    #
    # meanst = np.mean(mean_pre4ap, axis=1)
    # std = np.std(mean_pre4ap, axis=1, ddof=1)
    # ax.plot(range(len(meanst)), meanst, color='skyblue')
    # ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.3, color='skyblue')
    #
    # meanst = np.mean(mean_post4ap, axis=1)
    # std = np.std(mean_post4ap, axis=1, ddof=1)
    # ax.plot(range(len(meanst)), meanst, color='steelblue')
    # ax.fill_between(range(len(meanst)), meanst - std, meanst + std, alpha=0.3, color='steelblue')
    #
    # fig.show()

    # aoplot.plot_periphotostim_avg(arr=cell_avg_stim_traces,
    #                               figsize=[5, 4], title=('responses of photostim non-targets'),
    #                               y_label=y_label, x_label='Time post-stimulation (seconds)')

# %% 5.5.1) # # PLOT - average # of significant responders (+ve and -ve) for pre vs. post 4ap

# tips = sns.load_dataset("tips")

# sns.catplot(data=allopticalResults.num_sig_responders_df, order = ['pre4ap_pos', 'post4ap_pos', 'pre4ap_neg', 'post4ap_neg'])
# plt.show()


data=[]
cols = ['pre4ap_pos', 'post4ap_pos']
for col in cols:
    data.append(list(allopticalResults.num_sig_responders_df.loc[:, col]))

cols = ['pre4ap_neg', 'post4ap_neg']
for col in cols:
    data.append(list(allopticalResults.num_sig_responders_df.loc[:, col]))


experiments = ['RL108', 'RL109', 'PS05', 'PS07', 'PS06', 'PS11']
pre4ap_pos = []
pre4ap_neg = []
post4ap_pos = []
post4ap_neg = []

for exp in experiments:
    rows = []
    for row in range(len(allopticalResults.num_sig_responders_df.index)):
        if exp in allopticalResults.num_sig_responders_df.index[row]:
            rows.append(row)
    x = allopticalResults.num_sig_responders_df.iloc[rows, :].mean(axis=0)
    pre4ap_pos.append(round(x[0], 1))
    pre4ap_neg.append(round(x[1], 1))
    post4ap_pos.append(round(x[2], 1))
    post4ap_neg.append(round(x[3], 1))

data = [pre4ap_pos, post4ap_pos]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap_pos', 'post4ap_pos'], colors=['lightgreen', 'forestgreen'],
                        bar=True, paired=True, expand_size_x=0.6, expand_size_y=1.3, title='# of Positive responders',
                        y_label='# of sig. responders')

data = [pre4ap_neg, post4ap_neg]
pj.plot_bar_with_points(data, x_tick_labels=['pre4ap_neg', 'post4ap_neg'], colors=['skyblue', 'steelblue'],
                        bar=True, paired=True, expand_size_x=0.6, expand_size_y=1.3, title='# of Negative responders',
                        y_label= '# of sig. responders')





# %% 5.5.3) # # -  total post stim response evoked across all cells recorded
    # - like maybe add up all trials (sig and non sig), and all cells
    # - and compare pre-4ap and post-4ap (exp by exp, maybe normalizing the peak value per comparison from pre4ap?)
    # - or just make one graph per comparison and show all to Adam?


# %% 5.5.4) # # - think about some normalization via success rate of the stimulus (plot some response measure against success rate of the stimulation)
    # - calculate pearson's correlation value of the association



# %% 5.5) collect average stats for each prep, and summarize into the appropriate data point
num_sig_responders = pd.DataFrame(columns=['pre4ap_pos', 'pre4ap_neg', 'post4ap_pos', 'post4ap_neg'])
possig_responders_traces = []
negsig_responders_traces = []

allopticalResults.pre_stim_sec = 0.5
allopticalResults.stim_dur_sec = 0.25
allopticalResults.post_stim_sec = 3

for key in list(allopticalResults.trial_maps['pre'].keys()):
    name = allopticalResults.trial_maps['pre'][key][0][:-6]

    ########
    # pre-4ap trials calculations
    n_trials = len(allopticalResults.trial_maps['pre'][key])
    pre4ap_possig_responders_avgresponse = []
    pre4ap_negsig_responders_avgresponse = []
    pre4ap_num_pos = 0
    pre4ap_num_neg = 0
    for i in range(n_trials):
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, i))
        if sum(expobj.sig_units) > 0:
            name += expobj.metainfo['trial']
            pre4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            pre4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

            if i == 0:  # only taking trials averages from the first sub-trial from the matched comparisons
                pre4ap_possig_responders_avgresponse.append(pre4ap_possig_responders_avgresponse_)
                pre4ap_negsig_responders_avgresponse.append(pre4ap_negsig_responders_avgresponse_)
                stim_dur_fr = int(np.ceil(allopticalResults.stim_dur_sec * expobj.fps))  # setting 500ms as the dummy standardized stimduration
                pre_stim_fr = int(np.ceil(allopticalResults.pre_stim_sec * expobj.fps))  # setting the pre_stim array collection period again hard
                post_stim_fr = int(np.ceil(allopticalResults.post_stim_sec * expobj.fps))  # setting the post_stim array collection period again hard

                data_traces = []
                for trace in pre4ap_possig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[15] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                pre4ap_possig_responders_avgresponse = np.array(data_traces); print(f"shape of pre4ap_possig_responders array {pre4ap_possig_responders_avgresponse.shape} [5.5-1]")
                # print('stop here... [5.5-1]')

                data_traces = []
                for trace in pre4ap_negsig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[15] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                pre4ap_negsig_responders_avgresponse = np.array(data_traces); print(f"shape of pre4ap_negsig_responders array {pre4ap_negsig_responders_avgresponse.shape} [5.5-2]")
                # print('stop here... [5.5-2]')


                pre4ap_num_pos += len(pre4ap_possig_responders_avgresponse_)
                pre4ap_num_neg += len(pre4ap_negsig_responders_avgresponse_)
        else:
            pre4ap_num_pos += 0
            pre4ap_num_neg += 0
    pre4ap_num_pos = pre4ap_num_pos / n_trials
    pre4ap_num_neg = pre4ap_num_neg / n_trials


    # post-4ap trials calculations
    name += 'vs.'
    n_trials = len(allopticalResults.trial_maps['post'][key])
    post4ap_possig_responders_avgresponse = []
    post4ap_negsig_responders_avgresponse = []
    post4ap_num_pos = 0
    post4ap_num_neg = 0
    for i in range(n_trials):
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, i))
        if sum(expobj.sig_units) > 0:
            name += expobj.metainfo['trial']
            post4ap_possig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
            post4ap_negsig_responders_avgresponse_ = expobj.dfstdF_traces_avg[expobj.sig_units][
            np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

            if i == 0:
                post4ap_possig_responders_avgresponse.append(post4ap_possig_responders_avgresponse_)
                post4ap_negsig_responders_avgresponse.append(post4ap_negsig_responders_avgresponse_)
                stim_dur_fr = int(np.ceil(allopticalResults.stim_dur_sec * expobj.fps))  # setting 500ms as the dummy standardized stimduration
                pre_stim_fr = int(np.ceil(allopticalResults.pre_stim_sec * expobj.fps))  # setting the pre_stim array collection period again hard
                post_stim_fr = int(np.ceil(allopticalResults.post_stim_sec * expobj.fps))  # setting the post_stim array collection period again hard


                data_traces = []
                for trace in post4ap_possig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[15] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                post4ap_possig_responders_avgresponse = np.array(data_traces); print(f"shape of post4ap_possig_responders array {post4ap_possig_responders_avgresponse.shape} [5.5-3]")
                # print('stop here... [5.5-3]')


                data_traces = []
                for trace in post4ap_negsig_responders_avgresponse_:
                    trace_ = trace[expobj.pre_stim - pre_stim_fr : expobj.pre_stim]
                    trace_ = np.append(trace_, [[15] * stim_dur_fr])
                    trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames: expobj.pre_stim + expobj.stim_duration_frames + post_stim_fr])
                    data_traces.append(trace_)
                post4ap_negsig_responders_avgresponse = np.array(data_traces); print(f"shape of post4ap_negsig_responders array {post4ap_negsig_responders_avgresponse.shape} [5.5-4]")
                # print('stop here... [5.5-4]')


                post4ap_num_pos += len(post4ap_possig_responders_avgresponse_)
                post4ap_num_neg += len(post4ap_negsig_responders_avgresponse_)
        else:
            post4ap_num_pos += 0
            post4ap_num_neg += 0
    post4ap_num_pos = post4ap_num_pos / n_trials
    post4ap_num_neg = post4ap_num_neg / n_trials


    ########
    # place into the appropriate dataframe
    series = [pre4ap_num_pos, pre4ap_num_neg, post4ap_num_pos, post4ap_num_neg]
    num_sig_responders.loc[name] = series

    # place sig pos and sig neg traces into the list
    possig_responders_traces.append([pre4ap_possig_responders_avgresponse, post4ap_possig_responders_avgresponse])
    negsig_responders_traces.append([pre4ap_negsig_responders_avgresponse, post4ap_negsig_responders_avgresponse])

allopticalResults.num_sig_responders_df = num_sig_responders
allopticalResults.possig_responders_traces = np.asarray(possig_responders_traces)
allopticalResults.negsig_responders_traces = np.asarray(negsig_responders_traces)

allopticalResults.save()





# %% 5)  RUN DATA ANALYSIS OF NON TARGETS:
# #  - Analysis of responses of non-targets from suite2p ROIs in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
# #  - with option to plot only successful or only failure stims!

# import expobj
expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre g.1')
aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', skip_processing=False,
                                         save_plot_suffix='Nontargets_responses_2021-10-24/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))


# expobj.dff_traces, expobj.dff_traces_avg, expobj.dfstdF_traces, \
# expobj.dfstdF_traces_avg, expobj.raw_traces, expobj.raw_traces_avg = \
#     aoutils.get_nontargets_stim_traces_norm(expobj=expobj, normalize_to='pre-stim', pre_stim_sec=expobj.pre_stim_sec,
#                                             post_stim_sec=expobj.post_stim_sec)



# %% 5.4.1) for loop to go through each expobj to analyze nontargets
ls = ['PS05 t-010', 'PS06 t-011', 'PS11 t-010', 'PS17 t-005', 'PS17 t-006', 'PS17 t-007', 'PS18 t-006']
for key in list(allopticalResults.trial_maps['pre'].keys()):
    for j in range(len(allopticalResults.trial_maps['pre'][key])):
        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='pre %s.%s' % (key, j))
        if expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] in ls:
            aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', skip_processing=False,
                                                     save_plot_suffix='Nontargets_responses_2021-10-19/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        else:
            aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', skip_processing=True,
                                                     save_plot_suffix='Nontargets_responses_2021-10-19/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))

# %% 5.4.2) for loop to go through each expobj to analyze nontargets - post4ap trials
ls = pj.flattenx1(allopticalResults.post_4ap_trials)
for key in list(allopticalResults.trial_maps['post'].keys())[-5:]:
    for j in range(len(allopticalResults.trial_maps['post'][key])):
        # import expobj
        expobj, experiment = aoutils.import_expobj(aoresults_map_id='post %s.%s' % (key, j), do_processing=True)
        if expobj.metainfo['animal prep.'] + ' ' + expobj.metainfo['trial'] in ls:
            aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', skip_processing=False,
                                                     save_plot_suffix='Nontargets_responses_2021-10-24/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))
        else:
            aoutils.run_allopticalAnalysisNontargets(expobj, normalize_to='pre-stim', skip_processing=True,
                                                     save_plot_suffix='Nontargets_responses_2021-10-24/%s_%s.png' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial']))



# %% 5.2.1) plot dF/F of significant pos. and neg. responders that were derived from dF/stdF method
print('\n----------------------------------------------------------------')
print('plotting dFF for significant cells ')
print('----------------------------------------------------------------')

expobj.sig_cells = [expobj.s2p_cell_nontargets[i] for i, x in enumerate(expobj.sig_units) if x]
expobj.pos_sig_cells = [expobj.sig_cells[i] for i in np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
expobj.neg_sig_cells = [expobj.sig_cells[i] for i in np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
# plot peristim avg dFF of pos_sig_cells
selection = [expobj.s2p_cell_nontargets.index(i) for i in expobj.pos_sig_cells]
x = expobj.dff_traces_avg[selection]
y_label = 'pct. dFF (normalized to prestim period)'
f, ax[0, 0], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title='positive sig. responders', y_label=y_label, fig=f, ax=ax[0, 0], show=False,
                              x_label=None, y_lims=[-50, 200])

# plot peristim avg dFF of neg_sig_cells
selection = [expobj.s2p_cell_nontargets.index(i) for i in expobj.neg_sig_cells]
x = expobj.dff_traces_avg[selection]
y_label = None
f, ax[0, 1], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title='negative sig. responders', y_label=None, fig=f, ax=ax[0, 1], show=False,
                              x_label=None, y_lims=[-50, 200])

# plot peristim avg dFstdF of pos_sig_cells
selection = [expobj.s2p_cell_nontargets.index(i) for i in expobj.pos_sig_cells]
x = expobj.dfstdF_traces_avg[selection]
y_label = 'dF/stdF'
f, ax[1, 0], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title=None, y_label=y_label, fig=f, ax=ax[1, 0], show=False,
                              x_label='Time (seconds) ', y_lims=[-1, 1])

# plot peristim avg dFstdF of neg_sig_cells
selection = [expobj.s2p_cell_nontargets.index(i) for i in expobj.neg_sig_cells]
x = expobj.dfstdF_traces_avg[selection]
y_label = None
f, ax[1, 1], _ = aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3,
                              title=None, y_label=y_label, fig=f, ax=ax[1, 1], show=False,
                              x_label='Time (seconds) ', y_lims=[-1, 1])

f.show()




# %% 5.1) finding statistically significant followers responses
def _trialProcessing_nontargets(expobj):
    '''
    Uses dfstdf traces for individual cells and photostim trials, calculate the mean amplitudes of response and
    statistical significance across all trials for all cells

    Inputs:
        plane             - imaging plane n
    '''
    # make trial arrays from dff data shape: [cells x stims x frames]
    expobj._get_nontargets_stim_traces_norm(normalize_to='pre-stim', plot='dFstdF')

    # mean pre and post stimulus (within post-stim response window) flu trace values for all cells, all trials
    expobj.analysis_array = expobj.dfstdF_traces  # NOTE: USING dF/stdF TRACES
    expobj.pre_array = np.mean(expobj.analysis_array[:, :, expobj.pre_stim_frames_test], axis=1)  # [cells x prestim frames] (avg'd taken over all stims)
    expobj.post_array = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice], axis=1)  # [cells x poststim frames] (avg'd taken over all stims)

    # check if the two distributions of flu values (pre/post) are different
    assert expobj.pre_array.shape == expobj.post_array.shape, 'shapes for expobj.pre_array and expobj.post_array need to be the same for wilcoxon test'
    wilcoxons = np.empty(len(expobj.s2p_cell_nontargets))  # [cell (p-value)]

    for cell in range(len(expobj.s2p_cell_nontargets)):
        wilcoxons[cell] = stats.wilcoxon(expobj.post_array[cell], expobj.pre_array[cell])[1]

    expobj.wilcoxons = wilcoxons

    # ar2 = expobj.analysis_array[18, :, expobj.post_stim_frames_slice]
    # ar3 = ar2[~np.isnan(ar2).any(axis=1)]
    # assert np.nanmean(ar2) == np.nanmean(ar3)
    # expobj.analysis_array = expobj.analysis_array[~np.isnan(expobj.analysis_array).any(axis=1)]

    # measure avg response value for each trial, all cells --> return array with 3 axes [cells x response_magnitude_per_stim (avg'd taken over response window)]
    expobj.post_array_responses = []  ### this and the for loop below was implemented to try to root out stims with nan's but it's likley not necessary...
    for i in np.arange(expobj.analysis_array.shape[0]):
        a = expobj.analysis_array[i][~np.isnan(expobj.analysis_array[i]).any(axis=1)]
        responses = a.mean(axis=1)
        expobj.post_array_responses.append(responses)

    expobj.post_array_responses = np.mean(expobj.analysis_array[:, :, expobj.post_stim_frames_slice], axis=2)


    expobj.save()


def _sigTestAvgResponse_nontargets(expobj, alpha=0.1):
    """
    Uses the p values and a threshold for the Benjamini-Hochberg correction to return which
    cells are still significant after correcting for multiple significance testing
    """
    p_vals = expobj.wilcoxons
    expobj.sig_units = np.full_like(p_vals, False, dtype=bool)

    try:
        expobj.sig_units, _, _, _ = sm.stats.multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh',
                                                                     is_sorted=False, returnsorted=False)
    except ZeroDivisionError:
        print('no cells responding')

    # p values without bonferroni correction
    no_bonf_corr = [i for i, p in enumerate(p_vals) if p < 0.05]
    expobj.nomulti_sig_units = np.zeros(len(expobj.s2p_cell_nontargets), dtype='bool')
    expobj.nomulti_sig_units[no_bonf_corr] = True

    expobj.save()

    # p values after bonferroni correction
    #         bonf_corr = [i for i,p in enumerate(p_vals) if p < 0.05 / expobj.n_units[plane]]
    #         sig_units = np.zeros(expobj.n_units[plane], dtype='bool')
    #         sig_units[bonf_corr] = True






# %% 5.2) measuring/plotting responses of non targets to photostim - DEVELOPING CODE HERE

# xyloc plot of pos., neg. and non responders -- NOT SURE IF ITS WORKING PROPERLY RIGHT NOW, NOT WORTH THE EFFORT RIGHT NOW LIKE THIS. NOT THE FULL WAY TO MEASURE SPATIAL RELATIONSHIPS AT ALL AS WELL.
expobj.dfstdf_nontargets = pd.DataFrame(expobj.post_array_responses, index=expobj.s2p_cell_nontargets, columns=expobj.stim_start_frames)
df = pd.DataFrame(expobj.post_array_responses[expobj.sig_units, :], index=[expobj.s2p_cell_nontargets[i] for i, x in enumerate(expobj.sig_units) if x], columns=expobj.stim_start_frames)
s_ = np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)
df = pd.DataFrame(expobj.post_array_responses[s_, :][0], index=[expobj.s2p_cell_nontargets[i] for i in s_[0]], columns=expobj.stim_start_frames)
aoplot.xyloc_responses(expobj, df=df, clim=[-1, +1], plot_target_coords=True)


# TODO dynamic changes in responses across multiple stim trials - this is very similar to the deltaActivity measurements




# %% 5.3) creating large figures collating multiple plots describing responses of non targets to photostim for individual expobj's -- collecting code in aoutils.non_targets_responses()
plot_subset = False

if plot_subset:
    selection = np.random.randint(0, expobj.dff_traces_avg.shape[0], 100)
else:
    selection = np.arange(expobj.dff_traces_avg.shape[0])

#### SUITE2P NON-TARGETS - PLOTTING OF AVG PERI-PHOTOSTIM RESPONSES
f = plt.figure(figsize=[30, 10])
gs = f.add_gridspec(2, 9)

# %% 5.3.1) MAKE PLOT OF PERI-STIM AVG TRACES FOR ALL SIGNIFICANT AND NON-SIGNIFICANT RESPONDERS - also breaking down positive and negative responders

# PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
a1 = f.add_subplot(gs[0, 0:2])
x = expobj.dff_traces_avg[selection]
y_label = 'pct. dFF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=4,
                              title=None, y_label=y_label, fig=f, ax=a1, show=False,
                              x_label='Time (seconds)', y_lims=[-50, 200])
# PLOT AVG PHOTOSTIM PRE- POST- TRACE AVGed OVER ALL PHOTOSTIM. TRIALS
a2 = f.add_subplot(gs[0, 2:4])
x = expobj.dfstdF_traces_avg[selection]
y_label = 'dFstdF (normalized to prestim period)'
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=4,
                              title=None, y_label=y_label, fig=f, ax=a2, show=False,
                              x_label='Time (seconds)', y_lims=[-1, 3])
# PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/f
a3 = f.add_subplot(gs[0, 4:6])
vmin = -1
vmax = 1
aoplot.plot_traces_heatmap(expobj.dfstdF_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=int(1 * expobj.fps),
                           stim_off=int(1 * expobj.fps + expobj.stim_duration_frames - 1), xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                           title='dF/F heatmap for all nontargets', x_label='Time', cbar=True,
                           fig=f, ax=a3, show=False)
# PLOT HEATMAP OF AVG PRE- POST TRACE AVGed OVER ALL PHOTOSTIM. TRIALS - ALL CELLS (photostim targets at top) - Lloyd style :D - df/stdf
a4 = f.add_subplot(gs[0, -3:-1])
vmin = -100
vmax = 100
aoplot.plot_traces_heatmap(expobj.dff_traces_avg, expobj=expobj, vmin=vmin, vmax=vmax, stim_on=int(1 * expobj.fps),
                           stim_off=int(1 * expobj.fps + expobj.stim_duration_frames - 1), xlims=(0, expobj.dfstdF_traces_avg.shape[1]),
                           title='dF/stdF heatmap for all nontargets', x_label='Time', cbar=True,
                           fig=f, ax=a4, show=False)

# plot PERI-STIM AVG TRACES of sig nontargets
a10 = f.add_subplot(gs[1, 0:2])
x = expobj.dfstdF_traces_avg[expobj.sig_units]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a10, show=False,
                              title='significant responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of nonsig nontargets
a11 = f.add_subplot(gs[1, 2:4])
x = expobj.dfstdF_traces_avg[~expobj.sig_units]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a11, show=False,
                              title='non-significant responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of sig. positive responders
a12 = f.add_subplot(gs[1, 4:6])
x = expobj.dfstdF_traces_avg[expobj.sig_units][
    np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a12, show=False,
                              title='positive signif. responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# plot PERI-STIM AVG TRACES of sig. negative responders
a13 = f.add_subplot(gs[1, -3:-1])
x = expobj.dfstdF_traces_avg[expobj.sig_units][
    np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
aoplot.plot_periphotostim_avg(arr=x, expobj=expobj, pre_stim_sec=1, post_stim_sec=3, fig=f, ax=a13, show=False,
                              title='negative signif. responders', y_label='dFstdF (normalized to prestim period)',
                              x_label='Time (seconds)', y_lims=[-1, 3])

# %% 5.3.2) quantifying responses of non targets to photostim
# bar plot of avg post stim response quantified between responders and non-responders
a04 = f.add_subplot(gs[0, -1])
sig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)
nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
data = np.asarray([sig_responders_avgresponse, nonsig_responders_avgresponse])
pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'gray'], y_label='avg dF/stdF', bar=False,
                        text_list=['%s pct' % (np.round((len(sig_responders_avgresponse)/expobj.post_array_responses.shape[0]), 2) * 100),
                                   '%s pct' % (np.round((len(nonsig_responders_avgresponse)/expobj.post_array_responses.shape[0]), 2) * 100)],
                        text_y_pos=1.43, text_shift=1.7, x_tick_labels=['significant', 'non-significant'], expand_size_y=1.5, expand_size_x=0.6,
                        fig=f, ax=a04, show=False)


# bar plot of avg post stim response quantified between responders and non-responders
a14 = f.add_subplot(gs[1, -1])
possig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) > 0)[0]]
negsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1)[np.where(np.nanmean(expobj.post_array_responses[expobj.sig_units, :], axis=1) < 0)[0]]
nonsig_responders_avgresponse = np.nanmean(expobj.post_array_responses[~expobj.sig_units], axis=1)
data = np.asarray([possig_responders_avgresponse, negsig_responders_avgresponse, nonsig_responders_avgresponse])
pj.plot_bar_with_points(data=data, title='Avg stim response magnitude of cells', colors=['green', 'blue', 'gray'], y_label='avg dF/stdF', bar=False,
                        text_list=['%s pct' % (np.round((len(possig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1)),
                                   '%s pct' % (np.round((len(negsig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1)),
                                   '%s pct' % (np.round((len(nonsig_responders_avgresponse)/expobj.post_array_responses.shape[0]) * 100, 1))],
                        text_y_pos=1.43, text_shift=1.2, x_tick_labels=['pos. significant', 'neg. significant', 'non-significant'], expand_size_y=1.5, expand_size_x=0.5,
                        fig=f, ax=a14, show=False)

f.suptitle(
    ('%s %s %s' % (expobj.metainfo['animal prep.'], expobj.metainfo['trial'], expobj.metainfo['exptype'])))
f.show()






# %% 1) plot responses of SLM TARGETS in response to photostim trials - broken down by pre-4ap, outsz and insz (excl. sz bound)
# #  - with option to plot only successful or only failure stims!

### 1.1) PRE-4AP TRIALS
redo_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True  # avg only for each expobj
to_plot = 'successes'  # use for plotting either 'successes' stim responses or 'failures' stim responses

dffTraces = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.pre_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.pre_4ap_trials[0]
        # j = 0
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.1.0]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if redo_processing:
            aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                        plots=False, force_redo=False)
            expobj.save()


        if not hasattr(expobj, 'traces_SLMtargets_successes_avg'):
            print('running .calculate_SLMTarget_SuccessStims method for expobj of %s, %s [1.1.1]' % (prep, trial))
            expobj.stims_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stim_start_frames]
            expobj.StimSuccessRate_SLMtargets, expobj.traces_SLMtargets_successes_avg, \
            expobj.traces_SLMtargets_failures_avg = \
                expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets, stims_idx_l=expobj.stims_idx)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_successes_avg[key] for key in
                            expobj.traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.traces_SLMtargets_failures_avg[key] for key in
                                        expobj.traces_SLMtargets_failures_avg.keys()])

        # prepare data for plotting
        y_label = '% dFF (normalized to prestim period)'
        x_label = 'Time (secs)'
        pre_stim = 0.25; pre_stim_fr = expobj.fps * pre_stim
        post_stim = 2.75; post_stim_fr = expobj.fps * post_stim
        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 3 frames as stimduration
                trace_ = np.append(trace_, trace[expobj.pre_stim + expobj.stim_duration_frames:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - pre4ap stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - pre4ap stims %s %s' % (to_plot, prep, trial)
        # make plot
        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj, stim_duration=stim_dur, y_lims=[0, 50],
                                                 pre_stim_sec=0.25, post_stim_sec=2.75, avg_only=avg_only, title=title,
                                                 y_label=y_label, x_label=x_label, fig=f, ax=ax, show=False)


        print('|- shape of dFF array: ', data_traces.shape, ' [1.1.3]')
        print('exp_obj prestim / post stim: %s, %s' % (expobj.pre_stim/expobj.fps - 1/expobj.fps, expobj.pre_stim/expobj.fps + stim_dur))
        dffTraces.append(d[1:3])

f.show()
allopticalResults.dffTraces = np.asarray(dffTraces)
allopticalResults.save()





# %% ## 1.2) POST-4AP TRIALS - IN SZ STIMS - EXCLUDE STIMS/CELLS INSIDE SZ BOUNDARY
run_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True
to_plot = 'failures'  # use for plotting either 'successes' stim responses or 'failures' stim responses

dffTraces_insz = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.post_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[0]
        # j = 0
        # prep = 'RL109'
        # trial = 't-016'
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.2.1]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if 'post' in expobj.metainfo['exptype']:
            if run_processing:
                aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                            plots=False, force_redo=False)
            expobj.save()

        #### use expobj.hits_SLMtargets for determining which photostim trials to use - setting this up to only plot successfull trials
        if not hasattr(expobj, 'insz_traces_SLMtargets_successes_avg'):
            stims_insz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_in_sz]
            if len(stims_insz_idx) > 0:
                print('|- calculating stim success rates (insz) - %s stims [1.3.3]' % len(stims_insz_idx))
                expobj.insz_StimSuccessRate_SLMtargets, expobj.insz_traces_SLMtargets_successes_avg, \
                expobj.insz_traces_SLMtargets_failures_avg = \
                    expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets,
                                                            stims_idx_l=stims_insz_idx,
                                                            exclude_stims_targets=expobj.slmtargets_sz_stim)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_successes_avg[key] for key in expobj.insz_traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.insz_traces_SLMtargets_failures_avg[key] for key in expobj.insz_traces_SLMtargets_failures_avg.keys()])

        y_label = 'pct. dFF (normalized to prestim period)'

        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in array_to_plot:
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 5 frames as stimduration
                trace_ = np.append(trace_, trace[-expobj.post_stim:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - in sz stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - in sz stims %s %s' % (to_plot, prep, trial)

        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj,
                                                 stim_duration=stim_dur, y_lims=[0, 50], title=title, avg_only=avg_only,
                                                 pre_stim_sec=0.25, post_stim_sec=2.75,
                                                 y_label=y_label, x_label='Time (secs)', fig=f, ax=ax, show=False)

        print('|- shape of dFF array: ', data_traces.shape, ' [1.2.4]')
        dffTraces_insz.append(d)

f.show()
allopticalResults.dffTraces_outsz = np.asarray(dffTraces_insz)
allopticalResults.save()



# %% ## 1.3) POST-4AP TRIALS (OUT SZ STIMS)

run_processing = False  # flag to use when rerunning this whole for loop multiple times
avg_only = True
to_plot = 'successes'  # use for plotting either 'successes' stim responses or 'failures' stim responses
re_plot = True


dffTraces_outsz = []
f, ax = plt.subplots(figsize=[5, 4])
for i in allopticalResults.post_4ap_trials:
    for j in range(len(i)):
        # pass
        # i = allopticalResults.post_4ap_trials[1]
        # j = 0
        prep = i[j][:-6]
        trial = i[j][-5:]
        print('\nprogress @ ', prep, trial, ' [1.3.1]')
        expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

        if 'post' in expobj.metainfo['exptype']:
            if run_processing:
                aoutils.run_alloptical_processing_photostim(expobj, to_suite2p=expobj.suite2p_trials, baseline_trials=expobj.baseline_trials,
                                                            plots=False, force_redo=False)
                expobj.save()

        if not hasattr(expobj, 'outsz_traces_SLMtargets_successes_avg') or run_processing:
            stims_outsz_idx = [expobj.stim_start_frames.index(stim) for stim in expobj.stims_out_sz]
            if len(stims_outsz_idx) > 0:
                print('|- calculating stim success rates (outsz) - %s stims [1.3.3]' % len(stims_outsz_idx))
                expobj.outsz_StimSuccessRate_SLMtargets, expobj.outsz_traces_SLMtargets_successes_avg, \
                expobj.outsz_traces_SLMtargets_failures_avg = \
                    expobj.calculate_SLMTarget_SuccessStims(hits_df=expobj.hits_SLMtargets,
                                                            stims_idx_l=stims_outsz_idx)

        if to_plot == 'successes':
            array_to_plot = np.asarray([expobj.outsz_traces_SLMtargets_successes_avg[key] for key in expobj.outsz_traces_SLMtargets_successes_avg.keys()])
        elif to_plot == 'failures':
            array_to_plot = np.asarray([expobj.outsz_traces_SLMtargets_failures_avg[key] for key in expobj.outsz_traces_SLMtargets_failures_avg.keys()])

        y_label = 'pct. dFF (normalized to prestim period)'

        if avg_only:
            # modify matrix to exclude data from stim_dur period and replace with a flat line
            data_traces = []
            for trace in np.asarray([expobj.outsz_traces_SLMtargets_successes_avg[key] for key in expobj.outsz_traces_SLMtargets_successes_avg.keys()]):
                trace_ = trace[:expobj.pre_stim]
                trace_ = np.append(trace_, [[15]*3])  # setting 5 frames as stimduration
                trace_ = np.append(trace_, trace[-expobj.post_stim:])
                data_traces.append(trace_)
            data_traces = np.array(data_traces)
            stim_dur = 3 / expobj.fps
            title = '%s stims only, all exps. - avg. responses of photostim targets - out sz stims' % to_plot
        else:
            data_traces = array_to_plot
            stim_dur = expobj.stim_duration_frames / expobj.fps
            title = '%s stims only - avg. responses of photostim targets - out sz stims %s %s' % (to_plot, prep, trial)


        f, ax, d = aoplot.plot_periphotostim_avg(arr=data_traces, expobj=expobj,
                                                 stim_duration=stim_dur, y_lims=[0, 50],
                                                 pre_stim_sec=0.25, exp_prestim=expobj.pre_stim, post_stim_sec=2.75, avg_only=avg_only,
                                                 title=title, y_label=y_label, x_label='Time (secs)', fig=f, ax=ax, show=False)

        print('|- shape of dFF array: ', data_traces.shape, ' [1.3.4]')
        dffTraces_outsz.append(d)

f.show()
allopticalResults.dffTraces_outsz = np.asarray(dffTraces_outsz)
allopticalResults.save()






# %% 1.3.1)
from scipy.interpolate import interp1d

traces = []
x_long = allopticalResults.dffTraces_outsz[0][1]
f, ax = plt.subplots(figsize=(6, 5))
for trace in allopticalResults.dffTraces_outsz:
    if len(trace[1]) < len(x_long):
        f2 = interp1d(trace[1], trace[2])
        trace_plot = f2(x_long)
        ax.plot(x_long, trace_plot, color='gray')
    else:
        trace_plot = trace[2]
        ax.plot(trace[1], trace_plot, color='gray')
    traces.append(trace_plot)
ax.axvspan(0.4, 0.48 + 3 / 30, alpha=1, color='tomato', zorder=3)  # where 30 == fps for the fastest imaging experiments
avgTrace = np.mean(np.array(traces), axis=0)
ax.plot(x_long, avgTrace, color='black', lw=3)
ax.set_title('avg of all targets per exp. for stims out_sz - each trace = t-series from allopticalResults.post_4ap_trials - dFF photostim',
             horizontalalignment='center', verticalalignment='top', pad=35, fontsize=13, wrap=True)
ax.set_xlabel('Time (secs)')
ax.set_ylabel('dFF (norm. to pre-stim F)')
f.show()


# %% 1.4) COMPARISON OF RESPONSE MAGNITUDE OF SUCCESS STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.post_4ap_trials + allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_out_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (outsz) [1.4.2] ***********')
                    expobj.StimSuccessRate_SLMtargets_outsz, expobj.hits_SLMtargets_outsz, expobj.responses_SLMtargets_outsz, expobj.traces_SLMtargets_successes_outsz = \
                        expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_out_sz)
                    success_responses = expobj.hits_SLMtargets_outsz * expobj.responses_SLMtargets_outsz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response outsz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())

                # raw_traces_stims = expobj.SLMTargets_stims_raw[:, stims, :]
                if len(expobj.stims_in_sz) > 0:
                    print('\n Calculating stim success rates and response magnitudes (insz) [1.4.3] ***********')
                    expobj.StimSuccessRate_SLMtargets_insz, expobj.hits_SLMtargets_insz, expobj.responses_SLMtargets_insz, expobj.traces_SLMtargets_successes_insz = \
                        expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stims_in_sz)

                    success_responses = expobj.hits_SLMtargets_insz * expobj.responses_SLMtargets_insz
                    success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                    allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                        'prep_trial'] == i[j], 'mean dFF response insz (hits, all targets)'] = success_responses.mean()
                    print(success_responses.mean())


            elif 'pre' in expobj.metainfo['exptype']:
                seizure_filter = False
                print('\n Calculating stim success rates and response magnitudes [1.4.4] ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets, expobj.traces_SLMtargets_successes = \
                    expobj.calculate_SLMTarget_responses_dff(threshold=15, stims_to_use=expobj.stim_start_frames)

                success_responses = expobj.hits_SLMtargets * expobj.responses_SLMtargets
                success_responses = success_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response (hits, all targets)'] = success_responses.mean()
                print(success_responses.mean())

            expobj.save()
    allopticalResults.save()


## make bar plot using the collected response magnitudes
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response (hits, all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

outsz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response outsz (hits, all targets)'].values[0] for trial in i]
    outsz_response_magnitude.append(np.mean(x))

insz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response insz (hits, all targets)'].values[0] for trial in i]
    insz_response_magnitude.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_response_magnitude, outsz_response_magnitude, insz_response_magnitude], paired=True,
                        colors=['black', 'purple', 'red'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'outsz', 'insz'], title='Avg. Response magnitude of hits',
                        y_label='response magnitude (dFF)')


# %% 1.5) COMPARISON OF RESPONSE MAGNITUDE OF FAILURES STIMS. FROM PRE-4AP, OUT-SZ AND IN-SZ

run_processing = 0

## collecting the response magnitudes of success stims
if run_processing:
    for i in allopticalResults.pre_4ap_trials:
        for j in range(len(i)):
            prep = i[j][:-6]
            trial = i[j][-5:]
            print('\nprogress @ ', prep, trial, ' [1.4.1]')
            expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

            if 'post' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets_outsz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_outsz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response outsz (failures, all targets)'] = failure_responses.mean()

                inverse = (expobj.hits_SLMtargets_insz - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets_insz
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response insz (failures, all targets)'] = failure_responses.mean()

            elif 'pre' in expobj.metainfo['exptype']:
                inverse = (expobj.hits_SLMtargets - 1) * -1
                failure_responses = inverse * expobj.responses_SLMtargets
                failure_responses = failure_responses.replace(0, np.NaN).mean(axis=1)
                allopticalResults.slmtargets_stim_responses.loc[allopticalResults.slmtargets_stim_responses[
                                                                    'prep_trial'] == i[j], 'mean dFF response (failures, all targets)'] = failure_responses.mean()
    allopticalResults.save()


## make bar plot using the collected response magnitudes
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response (failures, all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

outsz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response outsz (failures, all targets)'].values[0] for trial in i]
    outsz_response_magnitude.append(np.mean(x))

insz_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean dFF response insz (failures, all targets)'].values[0] for trial in i]
    insz_response_magnitude.append(np.mean(x))


pj.plot_bar_with_points(data=[pre4ap_response_magnitude, outsz_response_magnitude, insz_response_magnitude], paired=False,
                        colors=['black', 'purple', 'red'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'outsz', 'insz'], title='Avg. Response magnitude of failures',
                        y_label='response magnitude (dFF)')








# %% 2) BAR PLOT FOR PHOTOSTIM RESPONSE MAGNITUDE B/W PRE AND POST 4AP TRIALS
pre4ap_response_magnitude = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    pre4ap_response_magnitude.append(np.mean(x))

post4ap_response_magnitude = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean response (dF/stdF all targets)'].values[0] for trial in i]
    post4ap_response_magnitude.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_response_magnitude, post4ap_response_magnitude], paired=True,
                        colors=['black', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'post-4ap'], title='Avg. Response magnitude',
                        y_label='response magnitude')

# %% 3) BAR PLOT FOR PHOTOSTIM RESPONSE RELIABILITY B/W PRE AND POST 4AP TRIALS
pre4ap_reliability = []
for i in allopticalResults.pre_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean reliability (>0.3 dF/stdF)'].values[0] for trial in i]
    pre4ap_reliability.append(np.mean(x))

post4ap_reliability = []
for i in allopticalResults.post_4ap_trials:
    x = [allopticalResults.slmtargets_stim_responses.loc[
             allopticalResults.slmtargets_stim_responses[
                 'prep_trial'] == trial, 'mean reliability (>0.3 dF/stdF)'].values[0] for trial in i]
    post4ap_reliability.append(np.mean(x))

pj.plot_bar_with_points(data=[pre4ap_reliability, post4ap_reliability], paired=True,
                        colors=['black', 'purple'], bar=False, expand_size_y=1.1, expand_size_x=0.6,
                        xlims=True, x_tick_labels=['pre-4ap', 'post-4ap'], title='Avg. Response Reliability',
                        y_label='% success rate of photostim')

# %% 4) plot peri-photostim avg traces for all trials analyzed to make sure they look alright
# -- plot as little postage stamps


# make one figure for each prep/trial (one little plot for each cell in that prep)
for exp in allopticalResults.pre_4ap_trials:
    # exp = ['RL108 t-009']
    calc_dff_stims = False
    for j in exp:
        if 'PS18' in j:
            date = allopticalResults.slmtargets_stim_responses.loc[
                allopticalResults.slmtargets_stim_responses['prep_trial'] == j, 'date'].values[0]
            pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
                date, j[:-6], date, j[-5:], date, j[-5:])  # specify path in Analysis folder to save pkl object

            expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)
            if calc_dff_stims:
                print('\n Calculating stim success rates and response magnitudes [4.1] ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=10,
                                                              stims_to_use=expobj.stim_start_frames)
                expobj.save()

            # raw_traces_stims = expobj.SLMTargets_stims_raw

            # expobj.post_stim_response_window_msec = 500
            # expobj.post_stim_response_frames_window = int(expobj.fps * expobj.post_stim_response_window_msec / 1000)

            nrows = expobj.n_targets_total // 4
            if expobj.n_targets_total % 4 > 0:
                nrows += 1
            ncols = 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3),
                                    constrained_layout=True)
            counter = 0
            axs[0, 0].set_xlabel('Frames')
            axs[0, 0].set_ylabel('% dFF')

            responses_magnitudes_successes = {}
            response_traces_successes = {}
            responses_magnitudes_failures = {}
            response_traces_failures = {}

            for cell in range(expobj.SLMTargets_stims_dff.shape[0]):
                a = counter // 4
                b = counter % 4
                print('\n%s' % counter)
                if cell not in responses_magnitudes_successes.keys():
                    responses_magnitudes_successes[cell] = []
                    response_traces_successes[cell] = np.zeros((expobj.SLMTargets_stims_dff.shape[-1]))
                    responses_magnitudes_failures[cell] = []
                    response_traces_failures[cell] = np.zeros((expobj.SLMTargets_stims_dff.shape[-1]))

                # reliability = expobj.StimSuccessRate_SLMtargets[cell]
                # f, axs = plt.subplots(figsize=(5, 12), nrows=3)
                # fig, ax = plt.subplots(figsize=(3, 3))

                success_stims = np.where(expobj.responses_SLMtargets.loc[cell] >= 0.1 * 100)
                fail_stims = np.where(expobj.responses_SLMtargets.loc[cell] < 0.1 * 100)
                for i in success_stims[0]:
                    trace = expobj.SLMTargets_stims_dff[cell][i]
                    axs[a, b].plot(trace, color='skyblue', zorder=2, alpha=0.05)

                for i in fail_stims[0]:
                    trace = expobj.SLMTargets_stims_dff[cell][i]
                    axs[a, b].plot(trace, color='gray', zorder=3, alpha=0.05)

                success_avg = np.nanmean(expobj.SLMTargets_stims_dff[cell][success_stims], axis=0)
                failures_avg = np.nanmean(expobj.SLMTargets_stims_dff[cell][fail_stims], axis=0)
                axs[a, b].plot(success_avg, color='navy', linewidth=2, zorder=4)
                axs[a, b].plot(failures_avg, color='black', linewidth=2, zorder=4)
                axs[a, b].set_ylim([-0.1 * 100, 0.6 * 100])
                axs[a, b].text(0.98, 0.97,
                               'Success rate: %s' % ('{:,.1f}'.format(expobj.StimSuccessRate_SLMtargets[cell])),
                               verticalalignment='top', horizontalalignment='right',
                               transform=axs[a, b].transAxes, fontweight='bold',
                               color='black')
                axs[a, b].margins(0)
                axs[a, b].axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, color='mistyrose',
                                  zorder=0)

                counter += 1
            fig.suptitle((str(exp) + ' %s - %s targets' % ('- values: pct. dff', len(expobj.SLMTargets_stims_dff))))
            fig.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s_individual targets dFF.png' % (
            date, j[:-6], date, j))
            fig.show()

        # for x in range(expobj.SLMTargets_stims_dff[cell].shape[0]):
        #     response = expobj.responses_SLMtargets.loc[cell, expobj.]
        #     trace = expobj.SLMTargets_stims_dff[cell][x]
        #
        #     response = np.mean(trace[expobj.pre_stim_sec + expobj.stim_duration_frames + 1:
        #                                          expobj.pre_stim_sec + expobj.stim_duration_frames +
        #                                          expobj.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
        #     if response >= 0.1*100:
        #         responses_magnitudes_successes[cell].append(round(response, 2))
        #         response_traces_successes[cell] = np.vstack((trace, response_traces_successes[cell]))
        #         axs[a, b].plot(trace, color='skyblue', zorder=2, alpha=0.05)
        #     else:
        #         responses_magnitudes_failures[cell].append(round(response, 2))
        #         response_traces_failures[cell] = np.vstack((trace, response_traces_failures[cell]))
        #         axs[a, b].plot(trace, color='gray', zorder=3, alpha=0.05)
        # make plot for each individual cell

        #     success_plots = np.nanmean(response_traces_successes[cell][:-1], axis=0)
        #     failures_plots = np.nanmean(response_traces_failures[cell][:-1], axis=0)
        #     axs[a, b].plot(success_plots, color='navy', linewidth=2, zorder=4)
        #     axs[a, b].plot(failures_plots, color='black', linewidth=2, zorder=4)
        #     axs[a, b].axvspan(expobj.pre_stim_sec, expobj.pre_stim_sec + expobj.stim_duration_frames, color='mistyrose',
        #                       zorder=0)
        #     # ax.plot(response_traces_failures[cell][1:], color='black', zorder=1, alpha=0.1)
        #     # ax.plot(np.mean(expobj.SLMTargets_stims_raw[0], axis=0), color='black', zorder=1)
        #     axs[a, b].set_ylim([-0.2*100, 1.2*100])
        #     axs[a, b].text(0.98, 0.97, 'Success rate: %s' % ('{:,.2f}'.format(
        #         len(responses_magnitudes_successes[cell]) / (
        #                     len(responses_magnitudes_failures[cell]) + len(responses_magnitudes_successes[cell])))),
        #                    verticalalignment='top', horizontalalignment='right',
        #                    transform=axs[a, b].transAxes, fontweight='bold',
        #                    color='black')
        #     counter += 1
        # fig.suptitle((str(i) + ' %s - %s targets' % ('- % dff', len(expobj.SLMTargets_stims_dff))), y=0.995)
        # plt.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s.png' % (date, j[:-6], date, j))
        # fig.show()

    #     for trace in raw_traces_stims[cell]:
    #         # calculate dFF (noramlized to pre-stim) for each trace
    #         # axs[0].plot(trace, color='black', alpha=0.1)
    #         pre_stim_mean = np.mean(trace[0:expobj.pre_stim_sec])
    #         response_trace = (trace - pre_stim_mean)
    #         response_trace1 = response_trace / pre_stim_mean
    #         # if np.nanmax(response_trace) > 1e100 and np.nanmin(response_trace) < -1e100:
    #         #     print('\n%s' % np.nanmean(response_trace))
    #         #     print(np.nanmax(response_trace))
    #         #     print(np.nanmin(response_trace))
    #         std_pre = np.std(trace[0:expobj.pre_stim_sec])
    #         response_trace2 = response_trace / std_pre
    #         measure = 'dF/F'
    #         to_plot = response_trace1
    #         # axs[1].plot(response_trace, color='green', alpha=0.1)
    #         # axs[2].plot(response_trace2, color='purple', alpha=0.1)
    #     # axs[2].axvspan(expobj.pre_stim_sec + expobj.stim_duration_frames, expobj.pre_stim_sec + expobj.stim_duration_frames + 1 + expobj.post_stim_response_frames_window, color='tomato')
    #         # response_trace = response_trace / std_pre
    #         # if dff_threshold:  # calculate dFF response for each stim trace
    #         #     response_trace = ((trace - pre_stim_mean)) #/ pre_stim_mean) * 100
    #         # else:  # calculate dF_stdF response for each stim trace
    #         #     pass
    #
    #         # calculate if the current trace beats the threshold for calculating reliability (note that this happens over a specific window just after the photostim)
    #         response = np.mean(to_plot[expobj.pre_stim_sec + expobj.stim_duration_frames + 1:
    #                                              expobj.pre_stim_sec + expobj.stim_duration_frames +
    #                                              expobj.post_stim_response_frames_window])  # calculate the dF over pre-stim mean F response within the response window
    #
    #         # response_result = response / std_pre  # normalize the delta F above pre-stim mean using std of the pre-stim
    #         # response_trace = response_trace / std_pre
    #         if response >= 0.2:
    #             responses_magnitudes_successes[cell].append(round(response, 2))
    #             response_traces_successes[cell] = np.vstack((to_plot, response_traces_successes[cell]))
    #             axs[a, b].plot(to_plot, color='seagreen', zorder=3, alpha=0.1)
    #         else:
    #             responses_magnitudes_failures[cell].append(round(response, 2))
    #             response_traces_failures[cell] = np.vstack((to_plot, response_traces_failures[cell]))
    #             axs[a, b].plot(to_plot, color='gray', zorder=2, alpha=0.1)
    #
    #     # make plot for each individual cell
    #     success_plots = np.nanmean(response_traces_successes[cell][:-1], axis=0)
    #     failures_plots = np.nanmean(response_traces_failures[cell][:-1], axis=0)
    #     axs[a, b].plot(success_plots, color='darkgreen', linewidth=2, zorder=4)
    #     axs[a, b].plot(failures_plots, color='black', linewidth=2, zorder=4)
    #     axs[a, b].axvspan(expobj.pre_stim_sec, expobj.pre_stim_sec + expobj.stim_duration_frames, color='mistyrose',
    #                       zorder=0)
    #     # ax.plot(response_traces_failures[cell][1:], color='black', zorder=1, alpha=0.1)
    #     # ax.plot(np.mean(expobj.SLMTargets_stims_raw[0], axis=0), color='black', zorder=1)
    #     axs[a, b].set_ylim([-0.2, 1.2])
    #     axs[a, b].text(0.98, 0.97, 'Success rate: %s' % ('{:,.2f}'.format(len(responses_magnitudes_successes[cell]) / (len(responses_magnitudes_failures[cell]) + len(responses_magnitudes_successes[cell])))),
    #             verticalalignment='top', horizontalalignment='right',
    #             transform=axs[a, b].transAxes, fontweight='bold',
    #             color='black')
    #     counter += 1
    # fig.suptitle((str(i) + ' %s - %s targets' % (measure, len(raw_traces_stims))), y=0.995)
    # plt.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s.png' % (date, j[:-6], date, j))
    # fig.show()

# for i in range(len(response_traces_successes[cell][:-1])):
#     plt.plot(response_traces_successes[cell][:-1][i])
# plt.show()

# for i in responses_magnitudes_successes.keys():
#     print(len(responses_magnitudes_successes))

# plot avg of successes in green
# plot avg of failures in gray
# plot line at dF_stdF = 0.3
# add text in plot for avg dF_stdF value of successes, and % of successes

# plot barplot with points only comparing response magnitude of successes

