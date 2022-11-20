
# Import required packages
import torch
import numpy as np
import normflows as nf

from sklearn.datasets import make_moons

from matplotlib import pyplot as plt

from tqdm import tqdm

import dataloader as dl
import common as com



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





if __name__ == "__main__":
    param = com.yaml_load()
    

    # Set up model

    # Define Gaussian base distribution
    base = nf.distributions.base.DiagGaussian(param["feature"]["n_mels"]*param["feature"]["frames"])
    # train_data.shape[1]
    # print(param["feature"]["n_mels"]*param["feature"]["frames"])
    # Define list of flows
    num_layers = param['NF_layers']
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 128 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([int(param["feature"]["n_mels"]*param["feature"]["frames"]/2), param["MLP_dim"], param["MLP_dim"], param["feature"]["n_mels"]*param["feature"]["frames"]], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)


    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    # device = torch.device('cpu')
    model = model.to(device)

    print(device)
    print(torch.cuda.get_device_name(0))

    # Plot target distribution
    
    train_data=dl.dataloader(machine_type='valve', begin=0, end=1000)
    # train_data/=1000
    # print(train_data)

    """
    Now, we are ready to train the flow model. This can be done in a similar fashion as standard neural networks. Since we use samples from the target for training, we use the forward KL divergence as objective, which is equivalent to maximum likelihood.
    """

    # Train model
    epochs = param["fit"]["epochs"]
    num_samples = len(train_data)
    # show_iter = 500
    batch_size = param["fit"]["batch_size"]

    loss_hist = np.array([])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
    
        # create mini-batch
        mini_batches = create_mini_batches(train_data, batch_size)
        for mini_batch in mini_batches[:-1]:
            # X_mini/=1000
            X_mini = torch.tensor(mini_batch).float().to(device)

            # Compute loss
            loss = model.forward_kld(X_mini)
            # print('l: ',loss)
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
        
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        print(' loss: ',loss_hist[-1])

    # Plot loss
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig('plots\\train_loss.png') 
    # plt.show()

    model.save('trained_model.pt')

