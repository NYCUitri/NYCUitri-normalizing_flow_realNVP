
import my_model
from matplotlib import pyplot as plt
import normflows as nf
import torch
import numpy as np
import json
import dataloader_test as dl_t
import common as com
import getThreshold as gt
from sklearn import metrics

def create_mini_batches(x, batch_size):
    mini_batches = []
    x = np.array(x)
    mini_batches = np.array_split(x, x.shape[0] // batch_size + 1)
    return mini_batches



'''load the traind model'''
param = com.yaml_load()
base = nf.distributions.base.DiagGaussian(param["feature"]["n_mels"]*param["feature"]["frames"])
num_layers = param['NF_layers']
flows = []
for i in range(num_layers):
    param_map = nf.nets.MLP([int(param["feature"]["n_mels"]*param["feature"]["frames"]/2), param["MLP_dim"], param["MLP_dim"], param["feature"]["n_mels"]*param["feature"]["frames"]], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode='swap'))

model = my_model.NormalizingFlow(base, flows)
model.load('trained_model_pump.pt')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

'''load the test data'''
test_data=dl_t.dataloader('pump', begin=0, end=200, get_label=True)
batch_size = param["test"]["batch_size"]
mini_batches = create_mini_batches(test_data, batch_size)
loss_test=np.array([])
'''run the model to get the loss'''
for mini_batch in mini_batches:
    X_mini = torch.tensor(mini_batch).float().to(device)
    
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()

    loss_test=np.concatenate((loss_test, loss), axis=None)

loss_upper = 1000
'''draw histogram'''
e=0
bins_list=[0]
i=0
# upper=int(max(loss_con))
loss_test[np.isnan(loss_test)] = loss_upper
loss_test[loss_test>loss_upper] = loss_upper
# loss_test/=loss_upper

print('max', max(loss_test))
print('min', min(loss_test))

d=max(loss_test)/2500


upper=max(loss_test)
while e<upper:
    e+=d
    bins_list.append(e)

fig, ax = plt.subplots(figsize =(10, 7))
draw=0
if draw:
    ax.hist(loss_test, bins = bins_list, color='orange', alpha=0.5)
    plt.show()
# print(len(loss_test))

with open('jsons\\labels_for_test.json') as jsonfile:
    labels = np.array(json.load(jsonfile))


auc = metrics.roc_auc_score(labels.flatten(), loss_test)
p_auc = metrics.roc_auc_score(labels.flatten(), loss_test, max_fpr=param["max_fpr"])

print('AUC score:', auc)
print('P_AUC score:', p_auc)