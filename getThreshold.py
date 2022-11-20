# read from json
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import scipy.stats 

class MyRandomVariableClass(stats.rv_continuous):
    # def __init__(self,  zeta=0.1, sigma=1, seed=None):
    def __init__(self,  seed=None):
        super().__init__(a=0,  seed=seed)

    def _cdf(self, x, zeta, sigma):

        return 1-(1+zeta/sigma*x)**(-1/zeta)



with open('loss_normal.json') as jsonfile:
    loss_normal = np.array(json.load(jsonfile))

with open('loss_anomaly.json') as jsonfile:
    loss_anomaly = np.array(json.load(jsonfile))

loss_con=np.concatenate((loss_normal, loss_anomaly), axis=None)

''' plot loss_normal and loss_anomaly'''
e=int(min(loss_con))
d=2.5
d=(int(max(loss_con))-int(min(loss_con)))/1000

bins_list=[e]
i=0
# upper=int(max(loss_con))
upper=max(loss_con)
# upper=50000
while e<upper:
    e+=d
    bins_list.append(e)
    # print(e)


draw=0
# print(bins_list)
if draw:
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(loss_normal, bins = bins_list, color='r', alpha=0.5)
    ax.hist(loss_anomaly, bins = bins_list, color='g', alpha=0.5)
    # ax.hist(loss_con, bins = bins_list, color='b', alpha=0.5)
# print(np.median(loss_normal))
# print(np.median(loss_anomaly))
    plt.show()


''' calculate threshold'''
p_unknown = 0.5
du = (np.max(loss_normal)-np.min(loss_anomaly))/10000
u = np.min(loss_anomaly)


# while u <= np.max(loss_normal):

#     u+=du
while u < np.max(loss_normal):
    loss_normal_fit = loss_normal[loss_normal>=u] - u
    loss_anomaly_fit = -(loss_anomaly[loss_anomaly<=u] - u)
    fit_rv_n=MyRandomVariableClass()
    zeta_fit_n, sigma_fit_n, loc1, scale1=(fit_rv_n.fit(loss_normal_fit, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    fit_rv_a=MyRandomVariableClass()
    zeta_fit_a, sigma_fit_a, loc1, scale1=(fit_rv_a.fit(loss_anomaly_fit, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    score = (1 - p_unknown) * 


# plot histogram of samples







fig, ax1 = plt.subplots()
# ax1.hist((loss_normal),bins=bins_list, color = 'b', alpha=0.5)
# ax1.hist((loss_normal[loss_normal>=u]),bins=bins_list, color = 'r', alpha=0.25)
# ax1.hist(loss_anomaly,bins=list(np.array(bins_list)), color = 'orange', alpha=0.25) #, weights= np.ones_like(loss_anomaly)/len(loss_anomaly)
# ax1.hist(loss_anomaly[loss_anomaly<=u],bins=list(np.array(bins_list)), color = 'purple', alpha=0.25) #, weights= np.ones_like(loss_anomaly_fit)/len(loss_anomaly_fit)
# ax1.hist(loss_anomaly_fit,bins=list(np.array(bins_list)-u), color = 'red', alpha=0.25) 


# plot PDF and CDF of distribution
pts = np.linspace(0, 5000, 10000)
# fig2, ax2 = plt.subplots(figsize =(10, 7))
# ax2 = ax1.twinx()
# ax2.set_ylim(0,1.1)


# print(zeta_fit_m, sigma_fit_m)
# ax1.plot(pts, fit_rv_n.pdf(pts,zeta=zeta_fit_m,sigma=sigma_fit_m), color='navy')

fig.tight_layout()

plt.show()
