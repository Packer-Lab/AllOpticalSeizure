import sys

from _utils_.io import import_expobj

sys.path.extend(['/home/pshah/Documents/code/AllOpticalSeizure', '/home/pshah/Documents/code/AllOpticalSeizure'])

import _alloptical_utils as Utils

from _analysis_._ClassExpSeizureAnalysis import ExpSeizureAnalysis as main
from _main_.Post4apMain import Post4ap

# expobj = import_expobj(exp_prep='RL108 t-009')

# %%

main.plot__photostim_timings_lfp(exp_prep='RL108 t-009', xlims=[188, 500], ylims=[-2, 7], color='black', linewidth = 0.4,
                                 marker_size=30)


# %% 0) initialize analysis module for each expobj


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=0, allow_rerun=1, run_trials=['RL108 t-011'])
def run__initExpSeizureAnalysis(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.ExpSeizure = main(expobj, not_flip_stims=None
                             )
    expobj.save()

@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=True,
                                run_trials=['RL108 t-011', 'RL108 t-013', 'RL109 t-020'])
def procedure__classifying_sz_boundary(**kwargs):
    """
    Full procedure for classifying targets (and eventually non-targets) as in or out of sz boundary for each stim.

    Procedure: Runs plotting of sz boundaries for all stims in sz, then asks for stims to correct classification as input,
    then runs plotting of sz boundaries again.

    Sz boundary is based on manual placement of two coordinates on the
    avg image of each stim frame. the purpose of this function is to review the classification for each stim during each seizure
    to see if the cells are classified correctly on either side of the boundary.

    :param kwargs:
    :return:
    """

    expobj: Post4ap = kwargs['expobj']


    # aoplot.plot_lfp_stims(expobj)

    # matlab_pairedmeasurements_path = '%s/paired_measurements/%s_%s_%s.mat' % (expobj.analysis_save_path[:-23], expobj.metainfo['date'], expobj.metainfo['animal prep.'], trial[2:])  # choose matlab path if need to use or use None for no additional bad frames
    # expobj.paqProcessing()
    # expobj.collect_seizures_info(seizures_lfp_timing_matarray=matlab_pairedmeasurements_path)
    # expobj.save()


    # ######## CLASSIFY SLM PHOTOSTIM TARGETS AS IN OR OUT OF current SZ location in the FOV
    # -- FIRST manually draw boundary on the image in ImageJ and save results as CSV to analysis folder under boundary_csv


    # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=True, save_img=True)
    # expobj.sz_locations_stims() #if not hasattr(expobj, 'stimsSzLocations') else None

    ######## - all stims in sz are classified, with individual sz events labelled
    expobj.ExpSeizure.classify_sz_boundaries_all_stims(expobj=expobj)

    expobj.save()


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=True,
                                run_trials=['PS04 t-018'])  # , 'RL109 t-017'])
def run__plot__sz_boundaries_all_stims(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.ExpSeizure.classify_sz_boundaries_all_stims(expobj=expobj)


@Utils.run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, allow_rerun=True,
                                run_trials=['RL108 t-011'])  # , 'RL109 t-017'])
def run__enter_input_stims_to_flip(expobj_r=None, **kwargs):
    expobj: Post4ap = kwargs['expobj'] if expobj_r is None else expobj_r
    print(expobj)
    main.enter_stims_to_flip(expobj=expobj)  # need to run this on the
    expobj.save()




def run_misc(expobj: Post4ap):
    # expobj.collect_seizures_info()
    # expobj.sz_locations_stims()
    main.plot__exp_sz_lfp_fov(expobj=expobj)
    # expobj.plot_SLMtargets_Locs(background=None)
    # expobj.avg_stim_images(stim_timings=expobj.stims_in_sz, peri_frames=50, to_plot=True, save_img=True)


# %% RUN PROCESSING CODE

if __name__ == '__main__':

    # expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-011')
    # run__initExpSeizureAnalysis()
    # run_misc(expobj)
    #
    # expobj: Post4ap = Utils.import_expobj(prep='RL109', trial='t-020')
    # run_misc(expobj)

    # expobj: Post4ap = Utils.import_expobj(prep='RL108', trial='t-013')
    # main.remove_stims_to_flip(expobj=expobj)

    # run__enter_input_stims_to_flip()

    procedure__classifying_sz_boundary()

    # main.FOVszInvasionTime()
    # main.plot__sz_incidence()
    # main.plot__sz_lengths()
    #
    # main.calc__szInvasionTime()
    # main.plot__sz_invasion()
    pass



# %% collecting not_flip_stims lists for experimemts
# 232 381 530 828 5440 5589 5738 5887 6036 6184 6333 6482 6631 6780 6928 7077 7226 7375 7524 7672 7821 7970 8119 8268 8416 8565 8714 8863 9160 13029 13178 13327 13476 13624 13773 13922 14071 14220 14368 14517 14666 14815 - RL109 t-020

# 1720 4092 4240 4388 7798 7946 8094 8242 11207 11355 11504 11652 11800 - RL108 t-011

# 476, 624, 772, 921, 1069, 1217, 1365, 1514, 1662, 1810, 1958, 4923, 5071, 5219, 6850, 6998, 8925, 9074, 9222, 9370,
# 9518, 12186, 12335, 12483, 12631, 13372, 13521  -- RL108 t-013
