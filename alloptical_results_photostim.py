# %% DATA ANALYSIS + PLOTTING FOR ALL-OPTICAL TWO-P PHOTOSTIM EXPERIMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
import utils.funcs_pj as pj

# import onePstim superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

# %% ########## BAR PLOT showing average success rate of photostimulation

trial = 't-010'
animal_prep = 'PS05'
date = '2021-01-08'
# IMPORT pkl file containing data in form of expobj
pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
    date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object

expobj, experiment = aoutils.import_expobj(pkl_path=pkl_path)

# plot across different groups
t009_pre_4ap_reliability = list(expobj.StimSuccessRate_SLMtargets.values())
# t011_post_4ap_reliabilty = list(expobj.StimSuccessRate_cells.values())  # reimport another expobj for post4ap trial
t013_post_4ap_reliabilty = list(expobj.StimSuccessRate_SLMtargets.values())  # reimport another expobj for post4ap trial
#
pj.plot_bar_with_points(data=[t009_pre_4ap_reliability, t013_post_4ap_reliabilty], xlims=[0.25, 0.3],
                        x_tick_labels=['pre-4ap', 'post-4ap'], colors=['green', 'deeppink'], y_label='% success stims.',
                        ylims=[0, 100], bar=False, title='success rate of stim. responses', expand_size_y=1.2,
                        expand_size_x=1.2)

# %% adding slm targets responses to alloptical results superobject.slmtargets_stim_responses

animal_prep = 'PS07'
date = '2021-01-19'
# trial = 't-009'

pre4ap_trials = ['t-007', 't-008', 't-009']
post4ap_trials = ['t-011', 't-016', 't-017']

# pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
#     date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object
#
# expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

counter = allopticalResults.slmtargets_stim_responses.shape[0] + 1
# counter = 6

for trial in pre4ap_trials + post4ap_trials:
    print(counter)
    pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (
        date, animal_prep, date, trial, date, trial)  # specify path in Analysis folder to save pkl object

    expobj, _ = aoutils.import_expobj(pkl_path=pkl_path)

    # add trials info to experiment
    expobj.metainfo['pre4ap_trials'] = pre4ap_trials
    expobj.metainfo['post4ap_trials'] = post4ap_trials
    expobj.save()

    # save to results object:
    allopticalResults.slmtargets_stim_responses.loc[counter, 'prep_trial'] = '%s %s' % (
        expobj.metainfo['animal prep.'], expobj.metainfo['trial'])
    allopticalResults.slmtargets_stim_responses.loc[counter, 'date'] = expobj.metainfo['date']
    allopticalResults.slmtargets_stim_responses.loc[counter, 'exptype'] = expobj.metainfo['exptype']
    if hasattr(expobj, 'stims_in_sz'):
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean(
            [[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean(
            [[np.mean(expobj.outsz_responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(
            list(expobj.outsz_StimSuccessRate_SLMtargets.values()))
    else:
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dF/stdF all targets)'] = np.mean(
            [[np.mean(expobj.responses_SLMtargets[i]) for i in range(expobj.n_targets_total)]])
        allopticalResults.slmtargets_stim_responses.loc[counter, 'mean reliability (>0.3 dF/stdF)'] = np.mean(
            list(expobj.StimSuccessRate_SLMtargets.values()))

    allopticalResults.slmtargets_stim_responses.loc[counter, 'mean response (dFF all targets)'] = np.nan
    counter += 1

allopticalResults.save()
allopticalResults.slmtargets_stim_responses

# %% comparing avg. response magnitudes for pre4ap and post4ap within same experiment prep.

allopticalResults.pre_4ap_trials = [
    ['RL108 t-009'],
    ['RL108 t-010'],
    ['RL109 t-007'],
    ['RL109 t-008'],
    ['RL109 t-013'],
    ['RL109 t-014'],
    ['PS04 t-012', 'PS04 t-014', 'PS04 t-017'],
    ['PS05 t-010'],
    ['PS07 t-007'],
    ['PS07 t-008'],
    ['PS07 t-009'],
    ['PS06 t-008', 'PS06 t-009', 'PS06 t-010'],
    ['PS06 t-011'],
    ['PS06 t-012'],
    ['PS11 t-007'],
    ['PS11 t-010'],
    ['PS17 t-005'],
    ['PS17 t-006', 'PS17 t-007'],
    ['PS18 t-006']
]

allopticalResults.post_4ap_trials = [
    ['RL108 t-013'],
    ['RL108 t-011'],
    ['RL109 t-020'],
    ['RL109 t-021'],
    ['RL109 t-018'],
    ['RL109 t-016', 'RL109 t-017'],
    ['PS04 t-018'],
    ['PS05 t-012'],
    ['PS07 t-011'],
    ['PS07 t-016'],
    ['PS07 t-017'],
    ['PS06 t-014', 'PS06 t-015'],
    ['PS06 t-013'],
    ['PS06 t-016'],
    ['PS11 t-016'],
    ['PS11 t-011'],
    ['PS17 t-011'],
    ['PS17 t-009'],
    ['PS18 t-008']
]
allopticalResults.save()


# %%
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
                        y_label='response (dF/prestim_stdF')

# %% plot peri-photostim avg traces for all trials analyzed to make sure they look alright
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
                print('\n Calculating stim success rates and response magnitudes ***********')
                expobj.StimSuccessRate_SLMtargets, expobj.hits_SLMtargets, expobj.responses_SLMtargets = \
                    aoutils.calculate_SLMTarget_responses_dff(expobj, threshold=10, stims_to_use=expobj.stim_start_frames)
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

                success_stims = np.where(expobj.responses_SLMtargets.loc[cell] >= 0.1*100)
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
                axs[a, b].set_ylim([-0.1*100, 0.6*100])
                axs[a, b].text(0.98, 0.97, 'Success rate: %s' % ('{:,.1f}'.format(expobj.StimSuccessRate_SLMtargets[cell])),
                               verticalalignment='top', horizontalalignment='right',
                               transform=axs[a, b].transAxes, fontweight='bold',
                               color='black')
                axs[a, b].margins(0)
                axs[a, b].axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, color='mistyrose',
                                  zorder=0)

                counter += 1
            fig.suptitle((str(exp) + ' %s - %s targets' % ('- values: pct. dff', len(expobj.SLMTargets_stims_dff))))
            fig.savefig('/home/pshah/mnt/qnap/Analysis/%s/%s/results/%s_%s_individual targets dFF.png' % (date, j[:-6], date, j))
            fig.show()

        # for x in range(expobj.SLMTargets_stims_dff[cell].shape[0]):
            #     response = expobj.responses_SLMtargets.loc[cell, expobj.]
            #     trace = expobj.SLMTargets_stims_dff[cell][x]
            #
            #     response = np.mean(trace[expobj.pre_stim + expobj.stim_duration_frames + 1:
            #                                          expobj.pre_stim + expobj.stim_duration_frames +
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
        #     axs[a, b].axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, color='mistyrose',
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
    #         pre_stim_mean = np.mean(trace[0:expobj.pre_stim])
    #         response_trace = (trace - pre_stim_mean)
    #         response_trace1 = response_trace / pre_stim_mean
    #         # if np.nanmax(response_trace) > 1e100 and np.nanmin(response_trace) < -1e100:
    #         #     print('\n%s' % np.nanmean(response_trace))
    #         #     print(np.nanmax(response_trace))
    #         #     print(np.nanmin(response_trace))
    #         std_pre = np.std(trace[0:expobj.pre_stim])
    #         response_trace2 = response_trace / std_pre
    #         measure = 'dF/F'
    #         to_plot = response_trace1
    #         # axs[1].plot(response_trace, color='green', alpha=0.1)
    #         # axs[2].plot(response_trace2, color='purple', alpha=0.1)
    #     # axs[2].axvspan(expobj.pre_stim + expobj.stim_duration_frames, expobj.pre_stim + expobj.stim_duration_frames + 1 + expobj.post_stim_response_frames_window, color='tomato')
    #         # response_trace = response_trace / std_pre
    #         # if dff_threshold:  # calculate dFF response for each stim trace
    #         #     response_trace = ((trace - pre_stim_mean)) #/ pre_stim_mean) * 100
    #         # else:  # calculate dF_stdF response for each stim trace
    #         #     pass
    #
    #         # calculate if the current trace beats the threshold for calculating reliability (note that this happens over a specific window just after the photostim)
    #         response = np.mean(to_plot[expobj.pre_stim + expobj.stim_duration_frames + 1:
    #                                              expobj.pre_stim + expobj.stim_duration_frames +
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
    #     axs[a, b].axvspan(expobj.pre_stim, expobj.pre_stim + expobj.stim_duration_frames, color='mistyrose',
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


# %%

for i in allopticalResults.post_4ap_trials:
#     if i = allopticalResults.post_4ap_trials[6]:
        for j in i:
            date = allopticalResults.slmtargets_stim_responses.loc[
                     allopticalResults.slmtargets_stim_responses[
                         'prep_trial'] == j, 'date'].values[0]
            trial = j[-5:]
            prep = j[:-6]
            pkl_path = "/home/pshah/mnt/qnap/Analysis/%s/%s/%s_%s/%s_%s.pkl" % (date, prep, date, trial, date, trial)
            expobj, experiment = aoutils.import_expobj(trial=trial, date=date, pkl_path=pkl_path, verbose=False)

        expobj.avg_stim_images(stim_timings=expobj.stim_start_frames, peri_frames=50, to_plot=True, save_img=True)