import numpy as np
from scipy import stats

gcb = np.load('./scorest-graphcodebert.npy')
cb = np.load('./scorest_codebert.npy')
label = np.load('./labels.npy')

y_preds_cb = (cb[:] > 0.99)
y_preds_gcb = (gcb[:] > 0.99)
y_preds_cb = y_preds_cb.astype(int)
y_preds_gcb = y_preds_gcb.astype(int)

res_cb = y_preds_cb[:] == label[:]
res_gcb = y_preds_gcb[:] == label[:]

# print(len(np.where(res[:] == 1)[0]))
# print(len(np.where(res[:] == 1)[0])/label.shape[0])

res1_cb = res_cb.reshape((456, -1)).mean(axis=1)  # *100
res1_gcb = res_gcb.reshape((456, -1)).mean(axis=1)  # *100
# print(res1_cb[:20])
# print(res1_gcb[:20])

# if pvalue>0.05,equal_val=Ture.
levene_res = stats.levene(res1_gcb, res1_cb)
print('levene', levene_res)

t_test = stats.ttest_ind(res1_gcb, res1_cb, equal_var=False)
print('t_test', t_test)

SP = np.sqrt(((res1_cb.shape[0]-1)*np.square(res1_cb.std())+(res1_gcb.shape[0]-1)*np.square(res1_gcb.std()))/(res1_cb.shape[0]+res1_gcb.shape[0]-2))
d = abs(res1_cb.mean()-res_gcb.mean())/SP
print('ES', d)

