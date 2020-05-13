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
from numpy import load
from torchsummary import summary

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Conv2d(256,512,2,stride = 1,padding = 0,bias = False),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 2,stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

generator = Generator()
g_opt = optim.Adam(generator.parameters(),lr = 0.0002,betas = (0.5,0.999))
discriminator = Discriminator()
d_opt = optim.Adam(discriminator.parameters(),lr = 0.0002,betas = (0.5,0.999))

criterion_GAN = nn.BCELoss()
criterion_pixelwise = nn.L1Loss()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
generator.to(dev)
discriminator.to(dev)

def load_real_samples(filename):
	# load compressed arrays
    data = load(filename)
	# unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X1 = (torch.Tensor(X1)).to(dev)
    X2 = (torch.Tensor(X2)).to(dev)
    print(X1.shape)
    return X1.permute(0,3,1,2), X2.permute(0,3,1,2)

def load_real_samples2(filename):
	# load compressed arrays
    data = load(filename)
	# unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return X1,X2

def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
    trainA, trainB = dataset
	# choose random instances
    ix = torch.randint(low = 0, high = trainA.shape[0], size = (n_samples,))
	# retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
    y = torch.ones((n_samples, patch_shape, patch_shape, 1)).to(dev)
    return X1, X2, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model(samples)
	# create 'fake' class labels (0)
	y = torch.zeros((len(X), patch_shape, patch_shape, 1)).to(dev)
	return X, y

def train_discriminator(X_realA,X_realB,X_fakeB,y_real):
    y_fake = torch.zeros((len(X_realA), 30,30, 1)).to(dev)
    d_opt.zero_grad()
    pred_real = discriminator(X_realA,X_realB)
    real_loss = criterion_GAN(pred_real,y_real)

    pred_fake = discriminator(X_realA,X_fakeB.detach())
    fake_loss = criterion_GAN(pred_fake,y_fake)

    total_loss = (real_loss + fake_loss)*0.5

    total_loss.backward()
    d_opt.step()
    return real_loss,fake_loss

def train_generator(X_realA,X_realB,X_fakeB,y_real,l):
    g_opt.zero_grad()
    pred_gen = discriminator(X_realA,X_fakeB)
    # print(pred_gen.shape)
    gan_loss = criterion_GAN(pred_gen,y_real)
    l1_loss = criterion_pixelwise(X_realB,X_fakeB)

    g_loss = gan_loss + l1_loss*l
    g_loss.backward()
    g_opt.step()
    return g_loss

def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
    X_realA, X_realB, _ = generate_real_samples(dataset, n_samples, 1)
    # X_realA2, X_realB2, _ = generate_real_samples(dataset2, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(generator, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    # X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        # print(X_realA2[i].shape)
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i].cpu().detach().permute(1,2,0))
	# plot generated target image
    for i in range(n_samples):
        # print(X_fakeB[i].shape)
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i].cpu().detach().permute(1,2,0))
	# plot real target imag
    for i in range(n_samples):
        # print(X_realB[i].shape)
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i].cpu().detach().permute(1,2,0))
	# save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
	# save the generator model
	# filename2 = 'model_%06d.h5' % (step+1)
	# g_model.save(filename2)
	# print('>Saved: %s and %s' % (filename1, filename2))

def train_gan(dataset,n_batch,n_patch,n_epochs = 100):
    trainA, trainB = dataset
    batches = int(len(trainA)/n_batch)
    for i in range(n_epochs*batches):
        X_realA,X_realB,y_real = generate_real_samples(dataset,n_batch,n_patch)
        # print(y_real.shape,)
        X_fakeB = generator(X_realA)
        # print(X_fakeB.shape)
        #TRAIN GENERATOR ON REAL
        g_loss = train_generator(X_realA,X_realB,X_fakeB,y_real,l = 100)
        #TRAIN DISCRIMINATOR ON SAMPLE OF FAKE AND REAL
        d_loss1,d_loss2 = train_discriminator(X_realA,X_realB,X_fakeB,y_real)
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
        if ((i+1) % (batches*5) == 0):
            summarize_performance(i, generator, dataset)

dataset = load_real_samples('maps_256.npz')
# dataset2 = load_real_samples2('maps_256.npz')
train_gan(dataset,1,30)
