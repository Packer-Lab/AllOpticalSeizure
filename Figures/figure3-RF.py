import sys

from _analysis_._ClassPhotostimAnalysisSlmTargets import PhotostimAnalysisSlmTargets
from _utils_.rfv_funcs import make_fig_layout, show_test_figure_layout, add_label_axes

sys.path.extend(['/home/pshah/Documents/code/reproducible_figures-main'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rep_fig_vis as rfv

# %%

## Set general plotting parameters
rfv.set_fontsize(7)

## Set parameters
n_cat = 2
n_misc_rows = 2
n_misc = 5
colour_list = ['#101820', '#1b362c', '#2f553d', '#4f7553', '#79936f', '#aeae92']
colours_misc_dict = {xx: colour_list[xx] for xx in range(len(colour_list))}

save_fig = True

## For this tutorial, I have made these sequential bools:
plot_axes = True
plot_content = True
plot_extra = True

np.random.seed(2)  # fix seed


# %% MAKING LAYOUT:

# panel_shape = ncols x nrows
# bound = l, t, r, b
# bound should be = b, t, l, r

layout = {
    'topleft': {'panel_shape': (1, 2),
                'bound': (0.05, 0.95, 0.25, 0.66)},
    'toprighttop': {'panel_shape': (n_cat, 1),
                 'bound': (0.35, 0.95, 0.95, 0.86)},
    'toprightbottom': {'panel_shape': (n_cat, 1),
               'bound': (0.35, 0.85, 0.95, 0.66)},
    'bottom': {'panel_shape': (4,1),
               'bound': (0.05, 0.60, 0.80, 0.50),
               'wspace': 0.6}
}

fig, axes, grid = rfv.make_fig_layout(layout=layout, dpi=100)

rfv.naked(axes['topleft'][0])
rfv.naked(axes['topleft'][1])
rfv.despine(axes['topleft'][1])

# rfv.show_test_figure_layout(fig, axes=axes)  # test what layout looks like quickly, but can also skip and moveon to plotting data.

# %% add plots to axes

main = PhotostimAnalysisSlmTargets
main.plot_photostim_traces_stacked_LFP_pre4ap_post4ap(ax_cat=(axes['toprighttop'], axes['toprightbottom']), fig=fig)

# %% add panel labels

ax=axes['topleft'][0]
rfv.add_label_axes(s='A', ax=ax)

ax=axes['toprighttop'][0]
rfv.add_label_axes(s='B', ax=ax)

ax=axes['toprightbottom'][0]
rfv.add_label_axes(s="B'", ax=ax, y_adjust=-0.01)

fig.show()

# %%

## First way of doing text, by plt.text()
axes['topleft'][0].text(s='A', x=-3.75, y=4.5,  # specify coords in data coord system of this ax
               fontdict={'weight': 'bold'})
rfv.show_test_figure_layout(fig, axes=axes)

ax_cat[0].text(s='C', x=-1.29, y=-5.4,  # specify coords in data coord system of this ax
               fontdict={'weight': 'bold'})

ax_im.text(s='MC Escher, 1948', x=np.mean(list(ax_im.get_xlim())), y=ax_im.get_ylim()[0] + 10,  # get ax limits to define coords
            fontdict={'ha': 'center', 'va': 'top',  # change text alignment to make centering easier
                      'style': 'italic'})

## Alternatively, use annotate to specificy coords in fraction of ax or fig
## (this is actually usually also easier to align panel labels)
ax_im.annotate(s='B', xy=(0.578, 0.965), xycoords='figure fraction',
               weight='bold')
ax_im.annotate(s='Some brownian motion examples', xy=(0.5, 0.39), xycoords='figure fraction',
               ha='center', weight='bold')



# %% ## Make figure: THIJS VERSION:
fig = plt.figure(constrained_layout=False,  # False better when customising grids
                 figsize=(8, 10), dpi=400)  # width x height in inches

## Make grids:
gs_topleft = fig.add_gridspec(ncols=1, nrows=1,
                               bottom=0.8, top=0.95, right=0.25,
                               left=0.05)  # leave a bit of space between grids (eg left here and right in grid above)

gs_toprighttop = fig.add_gridspec(ncols=n_cat, nrows=1,  # number of rows and columns
                              bottom=0.86, top=0.95, right=0.95, left=0.35,  # set bounds on 4 sides
                              wspace=0.1, hspace=0.4)  # width and height spacing between plot in this grid

gs_toprightbottom = fig.add_gridspec(ncols=n_cat, nrows=1,  # number of rows and columns
                              bottom=0.66, top=0.84, right=0.95, left=0.35,  # set bounds on 4 sides
                              wspace=0.1, hspace=0.4)  # width and height spacing between plot in this grid


# gs_bottom = fig.add_gridspec(ncols=n_misc, nrows=n_misc_rows,  # number of rows and columns
#                              bottom=0.08, top=0.35, right=0.95, left=0.05,  # set bounds on 4 sides
#                              wspace=0.2, hspace=0.6)  # width and height spacing between plot in this grid


## Create axes.

## Add image (top left):
ax_im = fig.add_subplot(gs_topleft[0])
rfv.naked(ax_im)

## Add 2 plots to top right bin:
ax_cat_top = {}
for ii in range(n_cat):
    ax_cat_top[ii] = fig.add_subplot(gs_toprighttop[ii])  # create ax by indexing grid object
    rfv.despine(ax_cat_top[ii])

ax_cat_bottom = {}
for ii in range(n_cat):
    ax_cat_bottom[ii] = fig.add_subplot(gs_toprightbottom[ii])  # create ax by indexing grid object
    rfv.despine(ax_cat_bottom[ii])

# ## Add more plots:
# ax_misc = {ii: {} for ii in range(n_misc_rows)}
#
# for ii in range(n_misc_rows):  # n rows
#     for jj in range(n_misc):
#         ax_misc[ii][jj] = fig.add_subplot(gs_bottom[ii, jj])  # 2D indexing because multiple rows & multiple columns
#         curr_ax = ax_misc[ii][jj]  # easier to type
#         rfv.despine(curr_ax)
#         if jj > 0:  # not left column
#             rfv.remove_yticklabels(curr_ax)

# fig.show()


# %% add plots to axes









# %%
## Add content to panels:
## Top left, let's draw some periodic functions:
for i_sin in range(6):
    curr_alpha = 1  # - i_sin / 7
    curr_colour = colours_misc_dict[i_sin]
    rfv.plot_sin_one_period(ax=ax_cat[0], phase=i_sin / 10,
                            alpha=curr_alpha, colour=curr_colour)
    rfv.plot_normal_distr(ax=ax_cat[1], std_distr=1 + 0.1 * i_sin,
                          alpha=curr_alpha, colour=curr_colour)

## Add image content
img = mpimg.imread('drawing-hands.jpg!Large.jpg')  # load image into memroy
ax_im.imshow(img,
             interpolation='none')  # generally it's best to disable interpolation (between neighbouring pixels)

## Add brownian motion
for i_misc in range(n_misc):
    curr_var = 1 + i_misc
    rfv.plot_brown_proc(ax_trace=ax_misc[0][i_misc], ax_hist=ax_misc[1][i_misc],
                        var=curr_var,
                        colour=colours_misc_dict[i_misc], plot_ylabel=(i_misc == 0))

    ax_misc[0][i_misc].annotate(s=f'Var = {curr_var}', xy=(0.04, 1), va='bottom',
                                xycoords='axes fraction', c=colours_misc_dict[i_misc])
for i_row in range(n_misc_rows):
    rfv.equal_lims_n_axs(ax_list=list(ax_misc[i_row].values()))

fig.align_ylabels(axs=[ax_cat[0], ax_cat[1], ax_misc[0][0], ax_misc[1][0]])




if plot_extra:
    ## First way of doing text, by plt.text()
    ax_cat[0].text(s='A', x=-1.29, y=1.25,  # specify coords in data coord system of this ax
                   fontdict={'weight': 'bold'})
    ax_cat[0].text(s='C', x=-1.29, y=-5.4,  # specify coords in data coord system of this ax
                   fontdict={'weight': 'bold'})

    ax_im.text(s='MC Escher, 1948', x=np.mean(list(ax_im.get_xlim())), y=ax_im.get_ylim()[0] + 10,
               # get ax limits to define coords
               fontdict={'ha': 'center', 'va': 'top',  # change text alignment to make centering easier
                         'style': 'italic'})

    ## Alternatively, use annotate to specificy coords in fraction of ax or fig
    ## (this is actually usually also easier to align panel labels)
    ax_im.annotate(s='B', xy=(0.578, 0.965), xycoords='figure fraction',
                   weight='bold')
    ax_im.annotate(s='Some brownian motion examples', xy=(0.5, 0.39), xycoords='figure fraction',
                   ha='center', weight='bold')

if save_fig:
    plt.savefig('Example_rep_fig.pdf', bbox_inches='tight')