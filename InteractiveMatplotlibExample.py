%matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle

#%%
trial = 't-008'
###### IMPORT pkl file containing exp_obj
pkl_path = "/Volumes/Extreme SSD/oxford-data/2020-03-19/2020-03-19_%s.pkl" % trial
with open(pkl_path, 'rb') as f:
    exp_obj = pickle.load(f)


#%% plot initial cell
cell = 30
idx = exp_obj.cell_id.index(cell)
fig = plt.figure()
x = range(exp_obj.raw[idx].shape[0]) # number of frames
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot(x, exp_obj.raw[idx]) # plot example bit of data in the fig

axcell = plt.axes([0.25, 0.1, 0.65, 0.03])
scell = Slider(axcell, 'Cell', 1, len(exp_obj.cell_id), valinit=idx, valstep=1)

def update(val):
    nidx = int(scell.val)
    line.set_ydata(exp_obj.raw[nidx]) # set the new ydata
    fig.canvas.draw_idle() # draw result on plot

scell.on_changed(update)
plt.show()

# interact(update,
#          trial=widgets.IntSlider(min=0, max=dfsf_trials.shape[2], step=1, value=0),
#          cell=widgets.IntSlider(min=0, max=dfsf_trials.shape[0], step=1, value=0)); # create sliders with predefined limits
