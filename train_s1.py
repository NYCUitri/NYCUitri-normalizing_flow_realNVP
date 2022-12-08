# Train the real nvp model

import my_model
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import dataloader as dl
import common as com


''' create mini batches '''
def create_mini_batches(x, batch_size):
    mini_batches = []
    np.random.shuffle(x)
    mini_batches = np.array_split(x, x.shape[0] // batch_size + 1)
    return mini_batches

if __name__ == "__main__":
    # load parameters from .yaml file
    param = com.yaml_load()
    
    ''' Set up model '''
    # Define Gaussian base distribution
    base = nf.distributions.base.DiagGaussian(param["feature"]["n_mels"]*param["feature"]["frames"])
    # Define list of flows
    num_layers = param['NF_layers']
    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([int(param["feature"]["n_mels"]*param["feature"]["frames"]/2), param["MLP_dim"], param["MLP_dim"], param["feature"]["n_mels"]*param["feature"]["frames"]], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(2, mode='swap'))
    model = my_model.NormalizingFlow(base, flows)


    # Move model on GPU 
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)
    print(device)
    print(torch.cuda.get_device_name(0))
    
    ''' load data '''
    train_data=dl.dataloader(machine_type='valve', begin=100, end=120)

    ''' Train model '''
    epochs = param["fit"]["epochs"]
    num_samples = len(train_data)
    batch_size = param["fit"]["batch_size_s1"]
    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-5)

    # If any errors occur(e.g. CUDA out of memory), the trained model will be saved as trained_model_error.pt.
    # If the training completes properly, the trained model will be saved as trained_model.pt.
    try:
        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
        
            # create mini-batch
            mini_batches = create_mini_batches(train_data, batch_size)
            for mini_batch in mini_batches:
                
                X_mini = torch.tensor(mini_batch).float().to(device)

                # Compute loss
                loss = model.forward_kld(X_mini)
                # print('l: ',loss)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
            
                # Log loss
                loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
            print(' loss: ',loss_hist[-1])
            # Plot loss
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.savefig('plots\\train_loss.png') 
        # plt.show()
        model.save('trained_model.pt')


    except Exception as e:
        print('error:', e)
        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.savefig('plots\\train_loss_error.png') 
        # plt.show()
        model.save('trained_model_error.pt')


