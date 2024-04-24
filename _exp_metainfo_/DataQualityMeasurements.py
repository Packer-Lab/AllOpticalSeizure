#%% import stuff

#%% count number of ROIs detected by suite2p before and after 4AP

suite2p_paths_pre4ap = []
suite2p_paths_post4ap = []

rois_pre4ap = []
rois_post4ap = []
for path in suite2p_paths_pre4ap:
    flu, _, _ = uf.s2p_loader(path, subtract_neuropil=True)
    rois_pre4ap.append(len(flu))

for path in suite2p_paths_post4ap:
    flu, _, _ = uf.s2p_loader(path, subtract_neuropil=True)
    rois_post4ap.append(len(flu))

# %% count number of good naparm responsive cells before and after 4AP

pkl_paths_pre4ap = []
pkl_paths_post4ap = []

photoresp_cells_pre4ap = []
photoresp_cells_post4ap = []
for path in pkl_paths_pre4ap:
    flu, _, _ = uf.s2p_loader(path, subtract_neuropil=True)
    photoresp_cells_pre4ap.append(len(flu))

for path in pkl_paths_post4ap:
    flu, _, _ = uf.s2p_loader(path, subtract_neuropil=True)
    photoresp_cells_post4ap.append(len(flu))