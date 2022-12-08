import my_model
import json
import normflows as nf
import torch
import numpy as np
import common as com
import dataloader_test as dl_t

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

'''load the threshold'''
with open('jsons\\threshold.json') as jsonfile:
    threshold = json.load(jsonfile)
'''load test data'''
test_data=dl_t.dataloader('valve', begin=0, end=200, get_label=False)
batch_size = param["fit"]["batch_size"]
mini_batches = create_mini_batches(test_data, batch_size)
pred_y=np.array([])
''' Run the model to get the loss. And determine whether the loss exceeds the threshold.'''
for mini_batch in mini_batches:
    X_mini = torch.tensor(mini_batch).float().to(device)
    z, loss_batch = model.x_to_z(X_mini)
    loss=loss_batch.to('cpu').detach().numpy()
    y_p = []
    for i in loss:
        if i > threshold:
            y_p.append(1)
        else:
            y_p.append(0)
    pred_y=np.concatenate((pred_y, np.array(y_p)), axis=None)



