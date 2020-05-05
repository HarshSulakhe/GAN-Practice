

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader,TensorDataset
from torch import nn,optim
from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np
# from torchsummary import summary

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 100
bs = 128

transform = transforms.Compose([
    transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root = '../data/CIFAR10',train = True, transform = transform,download = True)
test_dataset = datasets.CIFAR10(root = '../data/CIFAR10',train = False, transform = transform,download = False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = bs, shuffle = True,)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = bs, shuffle = False,)

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = torch.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.view(n_samples, latent_dim)
	return x_input,torch.zeros(n_samples)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,128,3,2,1)
        self.conv3 = nn.Conv2d(128,128,3,2,1)
        self.conv4 = nn.Conv2d(128,256,3,2,1)
        self.fc1 = nn.Linear(4096,1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.dropout(x.view(-1,4096),0.4)
        x = self.fc1(x)
        return torch.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, latent_dim,n_nodes):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, n_nodes)
        self.convt1 = nn.ConvTranspose2d(256,128,4,2,1)
        self.convt2 = nn.ConvTranspose2d(128,128,4,2,1)
        self.convt3 = nn.ConvTranspose2d(128,128,4,2,1)
        self.conv1 = nn.Conv2d(128,3,3,1,1)
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = x.view(-1,256,4,4)
        x = F.leaky_relu(self.convt1(x))
        x = F.leaky_relu(self.convt2(x))
        x = F.leaky_relu(self.convt3(x))
        x = self.conv1(x)
        return torch.tanh(x)

generator = Generator(LATENT_DIM,256*4*4)
discriminator = Discriminator()

discriminator.to(dev)
d_loss = nn.BCELoss()
d_opt = optim.Adam(discriminator.parameters(),lr = 0.0002,betas = (0.5,0.999))


generator.to(dev)
g_opt = optim.Adam(generator.parameters(),lr = 0.0002,betas = (0.5,0.999))


def train_discriminator(model,X_real,X_fake,loss,opt):
    opt.zero_grad()
    error_real = loss(model(X_real),torch.ones(X_real.size(0)).cuda())
    error_real.backward()
    error_fake = loss(model(X_fake),torch.zeros(X_fake.size(0)).cuda())
    error_fake.backward()
    opt.step()
    return error_fake+error_real

def train_generator(x_gen,d_model,opt,loss,n_samples = 256):
    opt.zero_grad()
    pred_samples = d_model(x_gen)
    # print(pred_samples[1])
    error = loss(pred_samples,torch.ones(n_samples).cuda())
    error.backward()
    opt.step()
    return error

def train_gan(g_model,d_model,loss,d_opt,g_opt,n_iter = 200,n_samples = 128):
    #TRAIN DIS
    for i in range(n_iter):
        for batch_idx, (x, _) in enumerate(train_loader):
            lat = generate_latent_points(LATENT_DIM,n_samples//2)[0].to(dev)
            X_fake = g_model(lat).detach()

            X_real = x.view(-1,3,32,32)
            X_real = Variable(X_real.to(dev))
            d_error = train_discriminator(d_model,X_real,X_fake,d_loss,d_opt,n_samples//2)
            #TRAIN GEN
            lat = generate_latent_points(LATENT_DIM,n_samples)[0].to(dev)
            x_gen = g_model(lat)
            g_error = train_generator(x_gen,d_model,g_opt,d_loss,n_samples)
            print(i,batch_idx,d_error.data.item(),g_error.data.item())
        if((i+1)%10==0):
            with torch.no_grad():
                x_progress = generate_latent_points(LATENT_DIM,25)[0].to(dev)
                x_creation = g_model(x_progress)
                x_creation = (x_creation + 1)/2
                for k in range(25):
                    plt.subplot(5,5,k+1)
                    plt.axis('off')
                    plt.imshow(x_creation[k].detach().cpu().permute(1,2,0))
                filename = '../data/CIFAR10/generated_plot_e%03d.png' % (i+1)
                plt.savefig(filename)
                plt.close()
                torch.save(g_model.state_dict(),'../data/CIFAR10/generator_{}'.format(i+1)+'.pth')
                # torch.save(d_model.state_dict(),'data/discriminator_{}'.format(i+1)+'.pth')

train_gan(generator,discriminator,d_loss,d_opt,g_opt)
