import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Autoencoder
from torchvision.utils import save_image
from utils import Hamiltonian

parser = argparse.ArgumentParser(description='Variational models')
# Path Arguments
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--input-size', type=int, default=28, metavar='N',
                    help='input space dimension')
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='hidden space dimension')
parser.add_argument('--output_size', type=int, default=32, metavar='N',
                    help='hidden space dimension')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='initialize seed')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='number of epochs')
parser.add_argument('--evaluate_interval', type=int, default=1, metavar='N',
                    help='number of epochs')
parser.add_argument('--hmc', type=bool, default=True, metavar='N',
                    help='number of epochs')
parser.add_argument('--num_steps_in_leap', type=int, default=10, metavar='N',
                    help='number of epochs')
parser.add_argument('--num_samples', type=int, default=5, metavar='N',
                    help='number of epochs')



args = parser.parse_args()

def get_loss(recon_x, x, mu, logvar):
    loss1 = criterion(recon_batch, data.view(-1, args.input_size*args.input_size))
    loss2 = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    loss2 /= data.size(0)*data.size(1)*data.size(2)
    loss3 = loss1+loss2
    return loss3

def evaluate_autoencoder(epoch):
    autoencoder.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data=data.cuda()
        data = Variable(data, volatile=True)
        reconst, mu, logvar, _ = autoencoder(data)
        test_loss += get_loss(reconst, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            samples = torch.cat([data[:n], reconst.view(-1, 1, args.input_size, args.input_size)[:n]])
            save_image(samples.data.cpu(), '/home/ddua/workspace/VAE/results_64_hmc/reconstruction_'+str(epoch)+'.png', nrow=n)
    
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss {0} <====='.format(test_loss))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)

autoencoder = Autoencoder(args.input_size, args.output_size, args.hidden_size, bn=False)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)
criterion = nn.BCELoss()
hmc = Hamiltonian(autoencoder.decoder, args.output_size, 0.1, args.num_steps_in_leap, args.num_samples)

if args.cuda:
    autoencoder = autoencoder.cuda()

for epoch in range(1, args.epochs + 1):
    autoencoder.train()
    train_loss = 0
    mnist_data = list(iter(train_loader))
    for batch_idx in range(0,1000):
        data = torch.FloatTensor(mnist_data[batch_idx][0])
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, encoded_rep = autoencoder(data)
        if args.hmc:
            init_x = encoded_rep.data.cpu().numpy()

        hmc.get_hmc_sample(init_x, data, gpu=args.cuda)
        #loss = get_loss(recon_batch, data, mu, logvar)
        #loss.backward()
        #train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
             print(batch_idx)
#             print('Train epoch : {} [{}/{} ({:.0f}%)]\tLoss {:.6f}'.format(epoch, 
#                         batch_idx * len(data), len(train_loader.dataset),
#                         (100 * batch_idx)/len(train_loader), loss.data[0]/len(data)))

    #print('=====> Epoch: {0} Average loss: {1} <===='.format(epoch, train_loss/len(train_loader.dataset)))
    if epoch % args.evaluate_interval == 0:
        print("evaluating model")
        evaluate_autoencoder(epoch)
        sample = Variable(torch.randn(1,args.output_size))
        if args.cuda:
            sample = sample.cuda()
        sample = autoencoder.decoder(sample).cpu()
        save_image(sample.data.view(-1, 1, args.input_size, args.input_size), \
                   'results_64_hmc/sample_'+str(epoch)+'.png')
        

