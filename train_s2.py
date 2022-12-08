# find the threshold

import my_model
from matplotlib import pyplot as plt
import normflows as nf
import torch
import numpy as np
import json
import dataloader as dl
import common as com
import getThreshold as gt

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
model.load('trained_model.pt')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


'''load normal data'''
normal_data=dl.dataloader(machine_type='pump', begin=0, end=20)
batch_size = param["fit"]["batch_size_s2"]
mini_batches = create_mini_batches(normal_data, batch_size)
loss_normal=np.array([])
''' Run the model to get the loss.'''
for mini_batch in mini_batches:
    X_mini = torch.tensor(mini_batch).float().to(device)
    
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()
    loss_normal=np.concatenate((loss_normal, loss), axis=None)

with open('jsons\\loss_normal.json', 'w', encoding='utf-8') as f:
    json.dump(list(loss_normal.astype(float)), f, ensure_ascii=False)


'''load outlier data'''
# Loading outlier data is kust for evt modeling. If get_threshold_simple() is chosen, this part is not needed.

# Note that dataloader_anomaly is used to load anomaly data. However, in the real training section,
# we might not have the anomaly data. So this dataloader_anomaly part is just for developing the program.

# import dataloader_anomaly
# outlier_data=dataloader_anomaly.dataloader(machine_type='pump', begin=0, end=20)

outlier_data=dl.dataloader(machine_type='ToyTrain', begin=100, end=105)
mini_batches = create_mini_batches(outlier_data, batch_size)
loss_anomaly=np.array([])
for mini_batch in mini_batches:
    X_mini = torch.tensor(mini_batch).float().to(device)
    
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()
    loss_anomaly=np.concatenate((loss_anomaly, loss), axis=None)
print('max normal:', max(loss_normal))
print('max anomaly:', max(loss_anomaly))
# ignore nan and too big loss
loss_anomaly = loss_anomaly[~np.isnan(loss_anomaly)]
loss_anomaly = loss_anomaly[loss_anomaly<1000]
loss_normal = loss_normal[loss_normal<1000]
with open('jsons\\loss_anomaly.json', 'w', encoding='utf-8') as f:
    json.dump(list(loss_anomaly.astype(float)), f, ensure_ascii=False)

'''draw histogram'''
loss_con=np.concatenate((loss_normal, loss_anomaly), axis=None)
e=int(min(loss_con))
# d=2.5
d=(int(max(loss_con))-int(min(loss_con)))/2500

bins_list=[e]
i=0
# upper=int(max(loss_con))
upper=max(loss_con)
print(max(loss_anomaly))
d = 2.5
upper=1000
while e<upper:
    e+=d
    bins_list.append(e)

fig, ax = plt.subplots(figsize =(10, 7))
draw=1
if draw:
    
    ax.hist(loss_normal, bins = bins_list, color='r', alpha=0.5)
    ax.hist(loss_anomaly, bins = bins_list, color='g', alpha=0.5)
    # ax.hist(loss_con, bins = bins_list, color='b', alpha=0.5)
print('median normal:', np.median(loss_normal))
print('median anomaly:', np.median(loss_anomaly))

plt.savefig('plots\\two_loss.png') 
plt.show()

# call get_threshold function
threshold = gt.get_threshold_simple()
# print("threshold:", threshold)
# Save the value to threshold.json.
with open('jsons\\threshold.json', 'w', encoding='utf-8') as f:
    json.dump(threshold, f, ensure_ascii=False)
