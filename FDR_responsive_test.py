# [cell x frame x trial]
all_trials = obj.all_trials[0]
​
# correcting for differences in trial length between trial types
if all_trials.shape[1] < spont_avg.shape[1]:
	spont_avg = spont_avg[:, :all_trials.shape[1]]
else:
	all_trials = all_trials[:, :spont_avg.shape[1], :]
​
if any(s in obj.stim_type for s in ['pr', 'ps']):
	subtracted_trials = all_trials-spont_avg[:, :, None]
​
# non-parametric test with repeated measures H_A = residual > | < 0 (post-pre)
pre_trial_frames = np.s_[obj.pre_frames - obj.test_frames : obj.pre_frames]
stim_end = obj.pre_frames + obj.duration_frames
post_trial_frames = np.s_[stim_end : stim_end + obj.test_frames]
​
pre_array = np.mean(subtracted_trials[:, pre_trial_frames, :], axis=1)
post_array = np.mean(subtracted_trials[:, post_trial_frames, :], axis=1)
​
wilcoxons = np.empty(obj.n_units[0])
​
for cell in range(obj.n_units[0]):
	wilcoxons[cell] = stats.wilcoxon(post_array[cell], pre_array[cell])[1]
​
# false discovery rate correction, 5% fixed FDR
p_value = wilcoxons
​
order_p = np.argsort(p_value) # index order
rank_p = np.argsort(order_p)+1 # sorted index order
​
n_tests = p_value.shape[0] # total number of Wilcoxon tests
fdr_rate = 0.05
​
q_value = (rank_p/n_tests)*fdr_rate
​
sorted_q = q_value[order_p]
sorted_p = p_value[order_p]
​
fdr_corr = sorted_p < sorted_q
final_idx = np.amax(np.argwhere(fdr_corr)) # the index of the final p_value in the ordered list that is less than the q_value
​
benhof_cells = order_p[:final_idx] # all p_values ranked lower (regardless of statistical significance) pass the correction
​
sig_cells = np.zeros(obj.n_units[0], dtype='bool')
sig_cells[benhof_cells] = True # creating new sig_cells array
​
s2_cells = obj.cell_s2[0] # boolean
​
print('Trial type:', obj.stim_type)
print('# sig. cell responses in S2',
	  '\nw/ FDR correction:', np.sum(sig_cells & s2_cells),
	  '\nw/ Bonferroni:', np.sum(obj.sta_sig[0][s2_cells]),
	  end='\n\n'
	 )