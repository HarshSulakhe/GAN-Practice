# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))])

train_dataset = datasets.CIFAR10(root = '../data/CIFAR10',train = True, transform = transform,download = True)
test_dataset = datasets.CIFAR10(root = '../data/CIFAR10',train = False, transform = transform,download = False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = bs, shuffle = True,)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = bs, shuffle = False,)

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

latent_dim = 100
cifar10_dim = 1024
G = Generator(latent_dim = latent_dim,n_nodes = 256*4*4).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

lr = 0.0002
G_optimizer = optim.Adam(G.parameters(),lr = lr,betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(),lr = lr,betas=(0.5,0.999))

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, 3,32,32), torch.ones(x.size(0), 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, latent_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs*2, latent_dim).to(device))
    y = Variable(torch.ones(bs*2, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

n_epoch = 200

for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        # D_losses.append(D_train(x))
        # G_losses.append(G_train(x))
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch), n_epoch, D_train(x), G_train(x)))
    if(epoch+1%10==0):
        with torch.no_grad():
            test_z = Variable(torch.randn(bs, latent_dim).to(device))
            generated = G(test_z)

            save_image(generated.view(generated.size(0), 1, 28, 28), '../data/CIFAR10/sample_{}'.format(epoch+1) + '.png')

# with torch.no_grad():
#     test_z = Variable(torch.randn(bs, latent_dim).to(device))
#     generated = G(test_z)
#
#     save_image(generated.view(generated.size(0), 1, 28, 28), './data/sample_' + '.png')
