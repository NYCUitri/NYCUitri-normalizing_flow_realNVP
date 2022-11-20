
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
    
    np.random.shuffle(x)

    n_minibatches = x.shape[0] // batch_size
    i = 0
 
    for i in range(n_minibatches + 1):
        mini_batch = x[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :]
        mini_batches.append((X_mini))

    if X_mini.shape[0] % batch_size != 0:
        mini_batch = x[i * batch_size:x.shape[0]]
        X_mini = mini_batch[:, :]
        mini_batches.append((X_mini))
    return mini_batches


param = com.yaml_load()


base = nf.distributions.base.DiagGaussian(param["feature"]["n_mels"]*param["feature"]["frames"])
num_layers = param['NF_layers']
flows = []
for i in range(num_layers):
    
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([int(param["feature"]["n_mels"]*param["feature"]["frames"]/2), param["MLP_dim"], param["MLP_dim"], param["feature"]["n_mels"]*param["feature"]["frames"]], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))


model = my_model.NormalizingFlow(base, flows)
model.load('trained_model.pt')
model.eval()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

normal_data=dl.dataloader('valve', begin=0, end=1000)
batch_size = param["fit"]["batch_size"]
mini_batches = create_mini_batches(normal_data, batch_size)
loss_normal=np.array([])
for mini_batch in mini_batches[:-1]:
    X_mini = torch.tensor(mini_batch).float().to(device)
    
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()
    loss_normal=np.concatenate((loss_normal, loss), axis=None)
# z=z.to('cpu').detach().numpy()






with open('loss_normal_1.json', 'w', encoding='utf-8') as f:
    json.dump(list(loss_normal.astype(float)), f, ensure_ascii=False)


# import dataloader_anomaly
# outlier_data=dataloader_anomaly.dataloader('valve', begin=0, end=1000)
outlier_data=dl.dataloader('ToyCar', begin=0, end=1000)


batch_size = param["fit"]["batch_size"]
mini_batches = create_mini_batches(outlier_data, batch_size)
loss_anomaly=np.array([])
for mini_batch in mini_batches[:-1]:
    X_mini = torch.tensor(mini_batch).float().to(device)
    
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()
    loss_anomaly=np.concatenate((loss_anomaly, loss), axis=None)


# remove nan and inf
loss_anomaly = loss_anomaly[~np.isnan(loss_anomaly)]

# loss_anomaly = loss_anomaly[~np.isinf(loss_anomaly)]
# u=10000
# loss_anomaly = loss_anomaly[loss_anomaly<u]
# new_anomaly=[]
# for e, l in zip(x_np, loss.to('cpu').detach().numpy()):
#     if l<u:
#         new_anomaly.append(e)
# new_anomaly=np.array(new_anomaly)
# sc.scatter(new_anomaly[:,0], new_anomaly[:,1], c='g', alpha=0.5)


with open('loss_anomaly_1.json', 'w', encoding='utf-8') as f:
    json.dump(list(loss_anomaly.astype(float)), f, ensure_ascii=False)






# draw histogram
loss_con=np.concatenate((loss_normal, loss_anomaly), axis=None)
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

fig, ax = plt.subplots(figsize =(10, 7))
draw=1
# print(bins_list)
if draw:
    
    ax.hist(loss_normal, bins = bins_list, color='r', alpha=0.5)
    ax.hist(loss_anomaly, bins = bins_list, color='g', alpha=0.5)
    # ax.hist(loss_con, bins = bins_list, color='b', alpha=0.5)
print(np.median(loss_normal))
print(np.median(loss_anomaly))

draw_anomaly=0
# print(int(min(loss_anomaly)))
# print(int(max(loss_anomaly)))
if draw_anomaly:
    
    e=int(min(loss_anomaly))
    d=(int(max(loss_anomaly))-int(min(loss_anomaly)))/100
    # d=10
    bins_list=[e]
    i=0
    upper=int(max(loss_anomaly))
    # upper=20
    while e<upper:
        e+=d
        bins_list.append(e)
        # print(e)
    ax.hist(loss_anomaly, bins = bins_list, color='g', alpha=0.5)

plt.savefig('plots\\two_loss_1.png') 
plt.show()




threshold = gt.get_threshold()
