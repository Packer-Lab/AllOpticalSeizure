import functools
import time
from funcsforprajay import funcs as pj

from _alloptical_utils import Utils
from _alloptical_utils import allopticalResults

## DECORATORS
def run_for_loop_across_exps(run_pre4ap_trials=False, run_post4ap_trials=False, skip_trials=[], run_trials=[]):
    """decorator to use for for-looping through experiment trials across run_pre4ap_trials and run_post4ap_trials.
    the trials to for loop through are defined in allopticalResults.pre_4ap_trials and allopticalResults.post_4ap_trials"""
    # if len(run_trials) > 0 or run_pre4ap_trials is True or run_post4ap_trials is True:
    print(f"\n {'..'*5} INITIATING FOR LOOP ACROSS EXPS {'..'*5}\n")
    t_start = time.time()
    def main_for_loop(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if run_trials:
                print(f"\n{'-' * 5} RUNNING SPECIFIED TRIALS from `trials_run` {'-' * 5}")
                counter1 = 0
                for i, exp_prep in enumerate(run_trials):
                    # print(i, exp_prep)
                    prep = exp_prep[:-6]
                    trial = exp_prep[-5:]
                    expobj, _ = Utils.import_expobj(prep=prep, trial=trial, verbose=False)

                    Utils.working_on(expobj)
                    func(expobj=expobj, **kwargs)
                    # try:
                    #     func(expobj=expobj, **kwargs)
                    # except:
                    #     print('Exception on the wrapped function call')
                    Utils.end_working_on(expobj)
                counter1 += 1


            if run_pre4ap_trials:
                print(f"\n{'-' * 5} RUNNING PRE4AP TRIALS {'-' * 5}")
                counter_i = 0
                res = []
                for i, x in enumerate(allopticalResults.pre_4ap_trials):
                    counter_j = 0
                    for j, exp_prep in enumerate(x):
                        if exp_prep in skip_trials:
                            pass
                        else:
                            # print(i, j, exp_prep)
                            prep = exp_prep[:-6]
                            pre4aptrial = exp_prep[-5:]
                            expobj, _ = Utils.import_expobj(prep=prep, trial=pre4aptrial, verbose=False)

                            Utils.working_on(expobj)
                            res_ = func(expobj=expobj, **kwargs)
                            # try:
                            #     func(expobj=expobj, **kwargs)
                            # except:
                            #     print('Exception on the wrapped function call')
                            Utils.end_working_on(expobj)
                            res.append(res_) if res_ is not None else None
                        counter_j += 1
                    counter_i += 1
                if res:
                    return res

            if run_post4ap_trials:
                print(f"\n{'-' * 5} RUNNING POST4AP TRIALS {'-' * 5}")
                counter_i = 0
                res = []
                for i, x in enumerate(allopticalResults.post_4ap_trials):
                    counter_j = 0
                    for j, exp_prep in enumerate(x):
                        if exp_prep in skip_trials:
                            pass
                        else:
                            # print(i, j, exp_prep)
                            prep = allopticalResults.post_4ap_trials[i][j][:-6]
                            post4aptrial = allopticalResults.post_4ap_trials[i][j][-5:]
                            try:
                                expobj, _ = Utils.import_expobj(prep=prep, trial=post4aptrial, verbose=False)
                            except:
                                raise ImportError(f"IMPORT ERROR IN {prep} {post4aptrial}")

                            Utils.working_on(expobj)
                            res_ = func(expobj=expobj, **kwargs)
                            # try:
                            #     func(expobj=expobj, **kwargs)
                            # except:
                            #     print('Exception on the wrapped function call')
                            Utils.end_working_on(expobj)
                            res.append(res_) if res_ is not None else None
                        counter_j += 1
                    counter_i += 1
                if res:
                    return res
            t_end = time.time()
            pj.timer(t_start, t_end)
            print(f" {'--' * 5} COMPLETED FOR LOOP ACROSS EXPS {'--' * 5}\n")
        return inner
    return main_for_loop

    # elif len(run_trials) > 0 and (run_pre4ap_trials is True or run_post4ap_trials is True):
    #     raise Exception('Cannot have both run_trials, and run_pre4ap_trials or run_post4ap_trials active on the same call.')
