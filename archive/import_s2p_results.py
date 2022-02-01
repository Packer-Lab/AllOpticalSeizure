import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/Vape')
import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_pj')
import _utils_.utils_funcs as uf #from Vape
import matplotlib.pyplot as plt
from _utils_ import funcs_pj as pjf

### import suite2p data
s2p_path = '/Volumes/Extreme SSD/oxford-data/2020-03-19/suite2p/photostim-4ap-t-017/plane0'
flu, spks, stat = uf.s2p_loader(s2p_path)

# dfof flu traces
flu_dff = []
for i in range(len(flu)):
    print(i)
    flu_dff.append(pjf.dff(flu[i]))


plt.figure(figsize=(20,3))
for i in s2p_cell_targets:
    plt.plot(flu[i], linewidth=0.1)
plt.show()

plt.figure(figsize=(20,3))
plt.plot(flu[28], linewidth=0.1); plt.show()

