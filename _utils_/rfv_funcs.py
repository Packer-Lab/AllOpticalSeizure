# mostly a copy of rfv project script --- used here just for testing things out..

from typing import Union

import numpy as np
from matplotlib import pyplot as plt

# %%

def add_scale_bar(ax, loc: tuple, length: tuple, bartype: str = "L", text: Union[str, tuple, list] = 'scalebar',
                  **kwargs):
    """
    Add a scale bar of the specified type to the ax object provided.

    :param ax:
    :param loc:
    :param length: length of scalebar line if L type: index 0 is the y scalebar and index 1 is the x scalebar.
    :param bartype:
    :param text: textlabel for scale bars. if L type: index 0 is the y scalebar and index 1 is the x scalebar.
    :param kwargs:
        text_offset: ratio to offset the text labels for the scalebars
        lw: linewidth of the scalebar
    """

    text_offset = [1] * len(text) if not 'text_offset' in kwargs else kwargs['text_offset']
    # text_offset_2 = [1] * len(text) if not 'text_offset_2' in kwargs else kwargs['text_offset_2']
    lw = 0.75 if not 'lw' in kwargs else kwargs['lw']
    fs = 10 if not 'fs' in kwargs else kwargs['fs']
    if bartype == 'L':
        ax.plot([loc[0]] * 2, [loc[1], loc[1] + length[0]], color='black', clip_on=False, lw=lw,
                solid_capstyle='butt')
        ax.plot((loc[0], loc[0] + length[1]), [loc[1]] * 2, color='black', clip_on=False, lw=lw,
                solid_capstyle='butt')
        # kwargs['fig'].show()
        assert type(text) is not str, 'incorrect type for L scalebar text provided.'
        assert len(text) == 2, 'L scalebar text argument must be of length: 2'

        ax.text(x=loc[0] - 1 * text_offset[0], y=loc[1], s=text[0], fontsize=fs, rotation=90)
        ax.text(x=loc[0], y=loc[1] - 1 * text_offset[1], s=text[1], fontsize=fs, rotation=0)

    else:
        raise ValueError(f'{type} not implemented currently.')


def make_fig_layout(layout: dict = None, **kwargs):
    pass

    """
    main idea is that the grid dictionary contains the necessary relationships for the layout.
    layout arg:
        # panel_shape = ncols x nrows
        # bound = l, t, r, b

    """

    figsize = (8, 10) if 'figsize' not in kwargs else kwargs['figsize']
    dpi = 400 if 'dpi' not in kwargs else kwargs['dpi']


    fig = plt.figure(constrained_layout=False,  # False better when customising grids
                     figsize=figsize, dpi=dpi)  # width x height in inches


    # layout = {
    #     'topleft': {
    #         'panel_shape': (1,2),
    #         'bound': (0.05, 0.95, 0.25, 0.8)}
    # }

    axes = {}  # this is the dictionary that will collect *all* axes that are required for this plot, named as per input grid

    for name, _grid in layout.items():
        wspace = 0.1 if 'wspace' not in _grid else _grid['wspace']
        hspace = 0.5 if 'hspace' not in _grid else _grid['hspace']

        gs_ = fig.add_gridspec(ncols=_grid['panel_shape'][0], nrows=_grid['panel_shape'][1],
                               left=_grid['bound'][0],
                               top=_grid['bound'][1],
                               right=_grid['bound'][2],
                               bottom=_grid['bound'][3],
                               wspace=wspace, hspace=hspace
                               )  # leave a bit of space between grids (eg left here and right in grid above)

        n_axs: int = _grid['panel_shape'][0] * _grid['panel_shape'][1]
        # _axes = {}
        if _grid['panel_shape'][0] > 1 and _grid['panel_shape'][1] > 1:
            _axes = np.empty(shape = (range(_grid['panel_shape'][0]), range(_grid['panel_shape'][1])), dtype=object)
            for col in range(_grid['panel_shape'][0]):
                for row in range(_grid['panel_shape'][1]):
                    _axes[col, row] = fig.add_subplot(gs_[col, row])  # create ax by indexing grid object

        elif _grid['panel_shape'][0] > 1 or _grid['panel_shape'][1] > 1:
            _axes = np.empty(shape = (n_axs), dtype=object)
            for i in range(n_axs):
                _axes[i] = fig.add_subplot(gs_[i])  # create ax by indexing grid object

        elif _grid['panel_shape'][0] == 1 or _grid['panel_shape'][1] == 1:
            pass
        else:
            pass

        axes[name] = _axes

    return fig, axes

def make_random_scatter(ax):
    ax.scatter(np.random.randn(100), np.random.randn(100), s=50, c='forestgreen')
    ax.set_ylabel('an y axis label')
    ax.set_xlabel('an x axis label')
    ax.set_title('an axis title')


def show_test_figure_layout(fig, axes):
    for grid, panels in axes.items():
        for ax in panels:
            make_random_scatter(ax)
    fig.show()


def add_label_axes(s, ax, **kwargs):
    """Add text annotation at xy coordinate (in units of figure fraction) to an axes object."""
    fs = 15 if 'fontsize' not in kwargs else kwargs['fontsize']
    y_adjust = 0.02 if 'y_adjust' not in kwargs else kwargs['y_adjust']
    x_adjust = 0.04 if 'x_adjust' not in kwargs else kwargs['x_adjust']
    if 'xy' not in kwargs:
        pos = np.array(ax.get_position())
        top = pos[1][1]
        left = pos[0][0]
        xy = (left - x_adjust, top + y_adjust)
    else:
        assert len(kwargs['xy']) == 2, 'xy coord length is not equal to 2.'
        xy = kwargs['xy']
    ax.annotate(s=s, xy=xy, xycoords='figure fraction', fontsize=fs, weight='bold')









