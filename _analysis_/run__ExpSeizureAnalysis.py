from typing import Union
import _alloptical_utils as Utils

from _analysis_._ClassExpSeizureAnalysis import ExpSeizureAnalysis as main


# %% 0) initialize analysis module for each expobj
from _main_.Post4apMain import Post4ap


@Utils.run_for_loop_across_exps(run_pre4ap_trials=0, run_post4ap_trials=1, allow_rerun=0)
def run__initExpSeizureAnalysis(**kwargs):
    expobj: Post4ap = kwargs['expobj']
    expobj.ExpSeizure = main(expobj)
    expobj.save()

# %%

if __name__ == '__main__':

    # run__initExpSeizureAnalysis()

    main.run__plot__sz_boundaries()
    #
    #
    # main.FOVszInvasionTime()
    # main.plot__sz_incidence()
    # main.plot__sz_lengths()
    #
    # main.calc__szInvasionTime()
    # main.plot__sz_invasion()

