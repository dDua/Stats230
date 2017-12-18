import torch.nn as nn
import torch
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, layers, bn=True):
        super(Encoder, self).__init__()

        self.layers = []

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            if bn:
                bn = nn.BatchNorm1d(layers[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            activation = nn.ReLU()
            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        self.init_weights()

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layers, bn=True):
        super(Decoder, self).__init__()
        
        self.layers = []

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            if bn:
                bn = nn.BatchNorm1d(layers[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            activation = nn.ReLU()
            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        self.init_weights()

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    

class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, bn=True):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        #self.layers = "400-20"
        self.encoder = Encoder([input_size*input_size, hidden_size, output_size], bn=bn)
        self.decoder = Decoder([output_size, hidden_size, input_size*input_size], bn=bn)
        self.linear_mu = nn.Linear(output_size, output_size)
        self.linear_var = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        input = input.view(-1, self.input_size*self.input_size)
        encoded = self.encoder(input)
        mu = self.linear_mu(encoded)
        logvar = self.linear_var(encoded)
        std = 0.5*logvar
        std = std.exp_()
        epsilon = Variable(std.data.new(std.size()).normal_())
        sample = mu + (epsilon * std)
        reconst = self.decoder(sample)
        reconst = self.sigmoid(reconst)
        return reconst, mu, logvar, sample

