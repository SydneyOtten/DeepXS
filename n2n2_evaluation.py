import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

fn_true = 'n2n2_NLO_y_test_data.txt'
fn_pred = 'n2n2_NLO_pred.txt'

raw_true = pd.read_csv(fn_true, header=None)
NLO_true = raw_true.values
NLO_true = NLO_true.astype(np.float)

raw_pred = pd.read_csv(fn_pred, header=None)
NLO_pred = raw_pred.values
NLO_pred = NLO_pred.astype(np.float)

percdev = (NLO_true - NLO_pred) / NLO_true

lim = 6.6*10**(-5)
percdev = (NLO_true - NLO_pred) / NLO_true
percdev_2 = percdev[NLO_true > lim]
N = percdev_2.shape[0]
one = np.int(0.6827*N)
two = np.int(0.9545*N)
three = np.int(0.9973*N)
test_mape = np.mean(np.abs(percdev_2))
print(test_mape)

eval = np.abs(percdev_2)
eval = np.sort(eval, axis=0)
one_sigma = eval[one]
two_sigma = eval[two]
three_sigma = eval[three]
print(one_sigma)
print(two_sigma)
print(three_sigma)


fig=plt.figure(figsize=(10, 10))
plt.plot(NLO_true, percdev, 'bo', alpha=0.35)
plt.title('NLO cross-section vs. relative error for $\chi^0_2/\chi^0_2$', fontsize=18)
plt.xlabel('$\sigma_{NLO}$[pb]', fontsize=18)
plt.ylabel('relative error $(\sigma_{true}-\sigma_{predicted})/\sigma_{true}$', fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('n2n2_NLO_rel_err.png')
plt.close()