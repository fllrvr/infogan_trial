import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, 
                 len_discrete_code, len_continuous_code):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # channel size of the output 
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code
        self.len_continuous_code = len_continuous_code
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code + self.len_continuous_code, 
                             1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Linear(1024, (self.input_size // 4) * (self.input_size // 4) * 128), 
            nn.BatchNorm1d((self.input_size // 4) * (self.input_size // 4) * 128), 
            nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=self.output_dim, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1))
        initialize_weights(self)
        
    def forward(self, input_, cont_code, disc_code):
        x = torch.cat([input_, cont_code, disc_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv1(x)
        x = self.deconv2(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, input_size,
                 len_discrete_code, len_continuous_code):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_disc = len_discrete_code
        self.len_conti = len_continuous_code
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1))
        self.fc = nn.Sequential(
                nn.Linear((self.input_size // 4) * (self.input_size // 4) * 128, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Linear(
                    1024,
                    self.output_dim + self.len_disc + 2 * self.len_conti))
        initialize_weights(self)
        
    def forward(self, input_):
        x = self.conv1(input_)
        x = self.conv2(x)
        x = x.view(-1, (self.input_size // 4) * (self.input_size // 4) * 128)
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim]).view(-1, 1)
        disc = x[:, self.output_dim:self.output_dim + self.len_disc]
        conti_mu = x[
            :,
            self.output_dim + self.len_disc:self.output_dim + self.len_disc + self.len_conti]
        conti_std_dev = torch.exp(x[:, self.output_dim + self.len_disc + self.len_conti:])
        
        return a, disc, conti_mu, conti_std_dev
