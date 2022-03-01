# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')


# %run ./setup_notebook.ipynb
# print(sys.path)

# IMPORT MODULES AND TRIAL expobj OBJECT

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
# sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils

import numpy as np
import matplotlib.pyplot as plt

# import results superobject that will collect analyses from various individual experiments
from _utils_.io import import_expobj

results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/'


########
#
# prep='RL108'
# trial='t-013'
# expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

# aoplot.plot_lfp_stims(expobj, xlims=[0.2e7, 1.0e7], linewidth=1.0)


# %% pickle cannot find attribute error
for i, x in enumerate(allopticalResults.post_4ap_trials):
    counter_j = 0
    for j, exp_prep in enumerate(x):
        prep = exp_prep[:-6]
        post4aptrial = exp_prep[-5:]
        try:
            expobj = import_expobj(prep=prep, trial=post4aptrial, verbose=False)
            print(f"loaded {expobj}")
        except:
            print(f'couldnt import expobj {exp_prep}')
        counter_j += 1


# %%

def smart_divide(func):
    def inner(**kwargs):
        print("I am going to divide", kwargs['a'], "and", kwargs['b'])
        kwargs['a'] = 20
        kwargs['b'] = 5

        if kwargs['b'] == 0:
            print("Whoops! cannot divide")
            return

        print(f"new kwargs {kwargs}")
        return func(**kwargs)
    return inner


@smart_divide
def divide(**kwargs):
    print(kwargs['a']/kwargs['b'])
    print(kwargs['c'])
divide(a=2, b=5, c=10)

# %% works
def fig_piping_decorator(func):
    def inner(*args, **kwargs):
        print(f'perform action 1')
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            if kwargs['fig'] is not None and kwargs['ax'] is not None:
                fig = kwargs['fig']
                ax = kwargs['ax']
        else:
            print('hello')
            if 'figsize' in kwargs.keys():
                kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=kwargs['figsize'])
            else:
                kwargs['fig'], kwargs['ax'] = plt.subplots()


        print(f"new kwargs {kwargs}")

        print(f'perform action 2')
        func(**kwargs)

        print(f'perform action 3')
        kwargs['fig'].suptitle('this title was decorated')
        if 'show' in kwargs.keys():
            if kwargs['show'] is True:
                fig.show()
            else:
                return fig, ax
        else:
            kwargs['fig'].show()

        return kwargs['fig'], kwargs['ax'] if 'fig' in kwargs.keys() else None

    return inner

@fig_piping_decorator
def make_plot(title='', **kwargs):
    kwargs['ax'].plot(np.arange(10))
    kwargs['ax'].set_title(title)

make_plot(title='a plot')


# %% works
def fig_piping_decorator(func):
    def inner(*args, **kwargs):
        print(f'perform action 1')
        print(f'original kwargs {kwargs}')
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            if kwargs['fig'] is not None and kwargs['ax'] is not None:
                fig = kwargs['fig']
                ax = kwargs['ax']
        else:
            print('making fig, ax')
            if 'figsize' in kwargs.keys():
                kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=kwargs['figsize'])
            else:
                kwargs['fig'], kwargs['ax'] = plt.subplots()


        print(f"new kwargs {kwargs}")

        print(f'perform action 2')
        func(**kwargs)

        print(f'perform action 3')
        kwargs['fig'].suptitle('this title was decorated')
        if 'show' in kwargs.keys():
            if kwargs['show'] is True:
                kwargs['fig'].show()
            else:
                return kwargs['fig'], kwargs['ax']
        else:
            kwargs['fig'].show()

        return kwargs['fig'], kwargs['ax'] if 'fig' in kwargs.keys() else None

    return inner

@fig_piping_decorator
def make_plot(title='', fig=None, ax=None, **kwargs):
    ax.plot(np.random.rand(10))
    ax.set_title(title)

fig, ax = make_plot(title='a plot', show=False)


# %% works
def plot_piping_decorator(plotting_func):
    def inner(**kwargs):
        print(f'perform action 1')
        print(f'original kwargs {kwargs}')
        return_fig_obj = False
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            if kwargs['fig'] is None or kwargs['ax'] is None:
                print('making fig, ax [1]')
                kwargs['fig'], kwargs['ax'] = plt.subplots()
            else:
                return_fig_obj = True
        else:
            if 'figsize' in kwargs.keys():
                kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=kwargs['figsize'])
            else:
                print('making fig, ax [2]')
                kwargs['fig'], kwargs['ax'] = plt.subplots()


        print(f"new kwargs {kwargs}")

        print(f'perform action 2')
        plotting_func(**kwargs)   # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()

        print(f'perform action 3')
        kwargs['fig'].suptitle('this title was decorated')
        if 'show' in kwargs.keys():
            if kwargs['show'] is True:
                print(f'showing fig...[3]')
                kwargs['fig'].show()
            else:
                print(f"value of return_fig_obj is {return_fig_obj} [4]")
                return (kwargs['fig'], kwargs['ax']) if return_fig_obj else None
        else:
            kwargs['fig'].show()

        print(f"value of return_fig_obj is {return_fig_obj} [5]")
        return (kwargs['fig'], kwargs['ax']) if return_fig_obj else None

    return inner

@fig_piping_decorator
def example_decorated_plot(title='', **kwargs):
    fig, ax = kwargs['fig'], kwargs['ax']
    print(f'kwargs inside example_decorated_plot definition: {kwargs}')
    ax.plot(np.random.rand(10))
    ax.set_title(title)


fig, ax = plt.subplots(figsize=(3,3))
fig, ax = example_decorated_plot(fig=fig, ax=ax, title='a plot', show=True)  # these are the original kwargs
# example_decorated_plot(title='a plot', show=True)  # these are the original kwargs
