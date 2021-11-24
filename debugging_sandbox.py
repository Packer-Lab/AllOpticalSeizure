# imports general modules, runs ipython magic commands
# change path in this notebook to point to repo locally
# n.b. sometimes need to run this cell twice to init the plotting paramters
# sys.path.append('/home/pshah/Documents/code/Vape/jupyter/')


# %run ./setup_notebook.ipynb
# print(sys.path)

# IMPORT MODULES AND TRIAL expobj OBJECT
import sys
import os

# sys.path.append('/home/pshah/Documents/code/PackerLab_pycharm/')
# sys.path.append('/home/pshah/Documents/code/')
import alloptical_utils_pj as aoutils
import alloptical_plotting_utils as aoplot
from funcsforprajay import funcs as pj

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numba import njit
from skimage import draw
import tifffile as tf

# import results superobject that will collect analyses from various individual experiments
results_object_path = '/home/pshah/mnt/qnap/Analysis/alloptical_results_superobject.pkl'
allopticalResults = aoutils.import_resultsobj(pkl_path=results_object_path)

save_path_prefix = '/home/pshah/mnt/qnap/Analysis/Results_figs/'


########
#
# prep='RL108'
# trial='t-013'
# expobj, experiment = aoutils.import_expobj(trial=trial, prep=prep, verbose=False)

# aoplot.plot_lfp_stims(expobj, xlims=[0.2e7, 1.0e7], linewidth=1.0)



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

make_plot(title='A plot')


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

fig, ax = make_plot(title='A plot', show=False)


# %% works
def fig_piping_decorator(func):
    def inner(**kwargs):
        print(f'perform action 1')
        print(f'original kwargs {kwargs}')
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            if kwargs['fig'] is not None and kwargs['ax'] is not None:
                fig = kwargs['fig']
                ax = kwargs['ax']
        else:
            if 'figsize' in kwargs.keys():
                fig, ax = plt.subplots(figsize=kwargs['figsize'])
            else:
                print('making fig, ax')
                fig, ax = plt.subplots()


        print(f"new kwargs {kwargs}")

        print(f'perform action 2')
        func(fig=fig, ax=ax, **kwargs)   # these are the original kwargs + any additional kwargs defined in inner()

        print(f'perform action 3')
        fig.suptitle('this title was decorated')
        if 'show' in kwargs.keys():
            if kwargs['show'] is True:
                fig.show()
            else:
                return fig, ax
        else:
            fig.show()

        return fig, ax if 'fig' in kwargs.keys() else None

    return inner

@fig_piping_decorator
def make_plot(fig, ax, title='', **kwargs):
    print(f'kwargs inside make_plot definition: {kwargs}')
    ax.plot(np.random.rand(10))
    ax.set_title(title)

fig, ax = make_plot(title='A plot', show=True)  # these are the original kwargs
