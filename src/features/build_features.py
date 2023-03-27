########################################################
### Sparse Autoencoder to extract nonlinear features ###
########################################################

'''
USAGE:
python sparse_ae_kl.py --epochs 10 --reg_param 0.001 --add_sparse yes
'''

import torch
import torchvision
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader,TensorDataset
from torchvision.utils import save_image
matplotlib.style.use('ggplot')

# constructing argument parsers 
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs to train our network for')
ap.add_argument('-l', '--reg_param', type=float, default=0.001, 
    help='regularization parameter `lambda`')
ap.add_argument('-sc', '--add_sparse', type=str, default='yes', 
    help='whether to add sparsity contraint or not')
args = vars(ap.parse_args())

EPOCHS = args['epochs']
BETA = args['reg_param']
ADD_SPARSITY = args['add_sparse']
RHO = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
print(f"Add sparsity regularization: {ADD_SPARSITY}")

# get the computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

'''
Need to get data into a DataLoader 
'''
i = 1
train_data = np.loadtxt("../../data/processed/standardized_train_FD00"+str(i)+".txt",delimiter=",")
test_data = np.loadtxt("../../data/processed/standardized_test_FD00"+str(i)+".txt",delimiter=",")
train_x = torch.Tensor(train_data[:,2:])
train_y = torch.Tensor(train_data[:,1])
test_x = torch.Tensor(test_data[:,2:])
test_y = torch.Tensor(test_data[:,1])

train_dataset = TensorDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset)
test_dataset = TensorDataset(test_x,test_y)
test_dataloader = DataLoader(test_dataset)

# define the autoencoder model
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=25, out_features=18)
        self.enc2 = nn.Linear(in_features=18, out_features=14)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=14, out_features=18)
        self.dec2 = nn.Linear(in_features=18, out_features=24)
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
 
        # decoding
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x
    
model = SparseAutoencoder().to(device)

# the loss function
criterion = nn.MSELoss()
# the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(F.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
# define the sparse loss function
def sparse_loss(rho, images):
    values = images
    loss = 0
    for i in range(len(model_children)):
        values = model_children[i](values)
        loss += kl_divergence(rho, values)
    return loss

# define the training function
def fit(model, dataloader, epoch):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(trainset)/dataloader.batch_size)):
        counter += 1
        img, _ = data
        img = img.to(device)
        img = img.view(img.size(0), -1)
        optimizer.zero_grad()
        outputs = model(img)
        mse_loss = criterion(outputs, img)
        if ADD_SPARSITY == 'yes':
            sparsity = sparse_loss(RHO, img)
            # add the sparsity penalty
            loss = mse_loss + BETA * sparsity
        else:
            loss = mse_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / counter
    print(f"Train Loss: {epoch_loss:.3f}")
    # save the reconstructed images 
    # save_decoded_image(outputs.cpu().data, f"../outputs/images/train{epoch}.png")
    return epoch_loss

# define the validation function
def validate(model, dataloader, epoch):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(testset)/dataloader.batch_size)):
            counter += 1
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            outputs = model(img)
            loss = criterion(outputs, img)
            running_loss += loss.item()
    epoch_loss = running_loss / counter
    print(f"Val Loss: {epoch_loss:.3f}")  
    # save the reconstructed images 
    outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
    save_image(outputs, f"../outputs/images/reconstruction{epoch}.png")
    return epoch_loss

# train and validate the autoencoder neural network
train_loss = []
val_loss = []
start = time.time()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} of {EPOCHS}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, testloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
end = time.time()
print(f"{(end-start)/60:.3} minutes")
# save the trained model
torch.save(model.state_dict(), f"../outputs/sparse_ae{EPOCHS}.pth")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()

