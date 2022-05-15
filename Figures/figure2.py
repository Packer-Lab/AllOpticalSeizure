import matplotlib as plt
import _utils_.alloptical_plotting as aoplot

# import onePstim superobject that will collect analyses from various individual experiments
from _exp_metainfo_.exp_metainfo import import_resultsobj, ExpMetainfo
from _utils_.io import import_expobj

results_object_path = '/home/pshah/mnt/qnap/Analysis/onePstim_results_superobject.pkl'
onePresults = import_resultsobj(pkl_path=results_object_path)


# %% B) LFP signal with optogenetic stims

date = '2021-01-24'

# pre4ap
expobj = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
                     ylims=[-4,1], xlims=[110*expobj.paq_rate, 210*expobj.paq_rate])


# post4ap
expobj = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial
aoplot.plotLfpSignal(expobj, x_axis='time', figsize=(5/100*150,3), linewidth=0.5, downsample=True, sz_markings=False, color='black',
                     ylims=[0,5], xlims=[10*expobj.paq_rate, 160*expobj.paq_rate])


# %% C) avg LFP trace 1p stim plots

date = '2021-01-24'

# pre4ap
pre4ap = import_expobj(prep='PS11', trial='t-009', date=date)  # pre4ap trial

assert 'pre' in pre4ap.exptype
aoplot.plot_flu_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue',
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Baseline', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(pre4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=pre4ap.stim_start_frames,
                                 title='Baseline')


# post4ap
post4ap = import_expobj(prep='PS11', trial='t-012', date=date)  # post4ap trial

assert 'post' in post4ap.exptype
aoplot.plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', stims_to_analyze=post4ap.stims_out_sz,
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Interictal', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=post4ap.stims_out_sz,
                                 title='Interictal')

aoplot.plot_flu_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, stim_span_color='skyblue', stims_to_analyze=post4ap.stims_in_sz,
                                 y_axis='dff', quantify=False, figsize=[3, 3], title='Ictal', ylims=[-0.5, 2.0])

aoplot.plot_lfp_1pstim_avg_trace(post4ap, x_axis='time', individual_traces=False, pre_stim=0.25, post_stim=0.75, write_full_text=False,
                                 optoloopback=True, figsize=(3.1, 3), shrink_text=0.8, stims_to_analyze=post4ap.stims_in_sz,
                                 title='Ictal')


# %% 1) Radial plot of Mean FOV with theta of










