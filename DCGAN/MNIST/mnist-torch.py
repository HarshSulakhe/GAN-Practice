

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn,optim
from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np
# from torchsummary import summary

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
LATENT_DIM = 100
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Lambda(nn.Module):

    def __init__(self,func):
        super().__init__()
        self.func = func

    def forward(self,x):
        return self.func(x)

def preprocess(x):
    return x.view(-1,1,28,28).to(dev)

def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = torch.randint(low = 0, high = dataset.size(0), size = (n_samples,1))
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = torch.ones((n_samples, 1))
	return X, y

# generate n fake samples with class labels
def generate_fake_samples(n_samples):
	# generate uniform random numbers in [0,1]
	X = torch.rand(28 * 28 * n_samples)
	# reshape into a batch of grayscale images
	X = X.view(n_samples,-1)
	# generate 'fake' class labels (0)
	y = torch.zeros((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = torch.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.view(n_samples, latent_dim)
	return x_input,torch.zeros(n_samples)


discriminator = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1,64,3,stride=2,padding=1),
    nn.LeakyReLU(negative_slope=0.2),
    nn.BatchNorm2d(64),
    nn.Dropout(0.4),
    nn.Conv2d(64,64,3,stride=2,padding=1),
    nn.LeakyReLU(negative_slope=0.2),
    nn.BatchNorm2d(64),
    nn.Dropout(0.4),
    Lambda(lambda x:x.view(x.size(0),-1)),
    nn.Linear(3136,1),
    nn.Sigmoid()
)

discriminator.to(dev)
d_loss = nn.BCELoss()
d_opt = optim.Adam(discriminator.parameters(),lr = 0.0002,betas = (0.5,0.999))

generator = nn.Sequential(
    Lambda(lambda x: x.to(dev)),
    nn.Linear(LATENT_DIM,128*7*7),
    nn.LeakyReLU(negative_slope=0.2),
    Lambda(lambda x: x.view(-1,128,7,7)),
    nn.ConvTranspose2d(128,128,4,2,padding = 1 ),
    nn.LeakyReLU(0.2),
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128,128,4,2,padding = 1),
    nn.LeakyReLU(0.2),
    nn.BatchNorm2d(128),
    nn.Conv2d(128,1,7,1,padding = 3),
    nn.Sigmoid()
)
generator.to(dev)
g_opt = optim.Adam(generator.parameters(),lr = 0.0002,betas = (0.5,0.999))

gan = nn.Sequential(
    generator,
    discriminator
)

def train_discriminator(model,X_real,X_fake,loss,opt, n_batch=256):
    opt.zero_grad()
    error_real = loss(model(X_real),torch.ones(n_batch).cuda())
    error_real.backward()
    error_fake = loss(model(X_fake),torch.zeros(n_batch).cuda())
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

def train_gan(g_model,d_model,loss,d_opt,g_opt,dataset,n_iter = 100,n_samples = 256):
    #TRAIN DIS
    for i in range(n_iter):
        for j in range(int(dataset.shape[0]/n_samples)):
            lat = generate_latent_points(LATENT_DIM,n_samples//2)[0]
            X_fake = g_model(lat).detach()

            X_real,_ = generate_real_samples(dataset,n_samples//2)
            d_error = train_discriminator(d_model,X_real,X_fake,d_loss,d_opt,n_samples//2)
            #TRAIN GEN
            lat = generate_latent_points(LATENT_DIM,n_samples)[0]
            x_gen = g_model(lat)
            g_error = train_generator(x_gen,d_model,g_opt,d_loss,n_samples)
        print(i,j)
        if(i%20==0):
            with torch.no_grad():
                x_progress = generate_latent_points(LATENT_DIM,25)[0]
                x_creation = g_model(x_progress)
                for k in range(25):
                    plt.subplot(5,5,k+1)
                    plt.axis('off')
                    plt.imshow(x_creation[k].detach().cpu().view(28,28),cmap = 'gray')
                filename = 'data/generated_plot_e%03d.png' % (i+1)
                plt.savefig(filename)
                plt.close()
                torch.save(g_model.state_dict(),'data/generator_{}'.format(i+1)+'.pth')
                torch.save(d_model.state_dict(),'data/discriminator_{}'.format(i+1)+'.pth')

train_gan(generator,discriminator,d_loss,d_opt,g_opt,x_train)
