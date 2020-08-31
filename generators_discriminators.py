import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Autoencoder import Encoder, Decoder, get_activation


class mlp_discriminator(nn.Module):
    def __init__(self, input_dim,
                 en_filters=[64, 128, 256, 256, 512],
                 de_filters=[128, 64, 1],
                 en_activation='relu', en_batch_norm=True,
                 de_activation='relu', de_batch_norm=False, de_batch_norm_last=False):
        super(mlp_discriminator, self).__init__()
        self.en_filters = en_filters
        self.de_filters = de_filters
        latent_dim = en_filters[-1]
        self.encoder =  Encoder(input_dim, en_filters, en_activation, en_batch_norm)
        self.decoder =  Decoder(latent_dim, de_filters, de_activation, de_batch_norm, de_batch_norm_last)
        self.fc = nn.Linear(latent_dim, 3)
    def forward(self, x):
        x = self.encoder(x)
        d_logit = self.decoder(x)
        d_prob = torch.sigmoid(d_logit)

        return d_logit, d_prob


class point_cloud_generator(nn.Module):
    def __init__(self, latent_dim, npoints, de_filters=[64, 128, 512, 1024],
                          de_activation='relu', de_batch_norm=False,
                          de_batch_norm_last=False):
        super(point_cloud_generator, self).__init__()
        self.decoder = Decoder(latent_dim, de_filters, de_activation, de_batch_norm)
        self.de_activation = de_activation
        self.de_batch_norm_last = de_batch_norm_last
        self.bn = nn.BatchNorm1d(de_filters[-1])
        self.npoints = npoints
        self.fc = nn.Linear(de_filters[-1], npoints * 3)
    def forward(self, x):
        output = self.decoder(x)
        activation = get_activation(self.de_activation)
        output = activation(output)
        #bn_size = output.size(1)

        if self.de_batch_norm_last:
            #bn = nn.BatchNorm1d(bn_size)
            #output = bn(output)
            output = self.bn(output)

        #ln = nn.Linear(bn_size, npoints * 3)
        #output = ln(output)
        output = self.fc(output)
        output = output.reshape(-1, 3, self.npoints)
        return output

class latent_code_discriminator_twp_layers(nn.Module):
    def __init__(self, latent_dim, de_filters=[256, 512], de_batch_norm=False,
                 de_activation='relu'):
        super(latent_code_discriminator_twp_layers, self).__init__()
        de_filters = de_filters + [1]
        self.decoder = Decoder(latent_dim, de_filters, de_activation, de_batch_norm)

    def forward(self, x):
        d_logit = self.decoder(x)
        d_prob = torch.sigmoid(d_logit)
        return d_logit, d_prob

class latent_code_generator_two_layers(nn.Module):
    def __init__(self, latent_dim, out_dim, de_filters=[128],
                 de_activation='relu', de_batch_norm=False):
        super(latent_code_generator_two_layers, self).__init__()
        de_filters = de_filters + out_dim
        self.decoder = Decoder(latent_dim, de_filters, de_activation, de_batch_norm)

    def forward(self, x):
        output = self.decoder(x)
        output = F.relu(output)
        return output

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1), dtype=torch.float32)
    #alpha = alpha.expand(real_samples.size())
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = (alpha * real_samples) + ((1 - alpha) * fake_samples)
    interpolates = interpolates.cuda() if torch.cuda.is_available() else interpolates
    interpolates = Variable(interpolates, requires_grad=True)

    interpolates_logit, _ = discriminator(interpolates)

        #fake = autograd.Variable((real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(interpolates_logit.size()).cuda() if torch.cuda.is_available() else torch.ones(
        interpolates_logit.size())
    gradients = torch.autograd.grad(
            outputs=interpolates_logit,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

'''
def mlp_discriminator(input, en_activation='relu', en_batch_norm=True,
                      de_activation='relu', de_batch_norm=False, de_batch_norm_last=False):
    en_filters = [64, 128, 256, 256, 512]
    de_filters = [128, 64, 1]
    input_dim = input.size(1)       #3
    num_en_filters = len(en_filters)
    num_de_filters = len(de_filters)
    latent_dim = en_filters[num_en_filters - 1]
    encoder = Encoder(input_dim, en_filters, en_activation, en_batch_norm)
    decoder = Decoder(latent_dim, de_filters, de_activation, de_batch_norm, de_batch_norm_last)

    x = encoder(input)
    d_logit = decoder(x)
    d_prob = torch.sigmoid(d_logit)

    return d_logit, d_prob

def point_cloud_generator(z, npoints, de_filters=[64, 128, 512, 1024],
                          de_activation='relu', de_batch_norm=False,
                          de_batch_norm_last=False):
    latent_dim = z.size(1)
    decoder = Decoder(latent_dim, de_filters, de_activation, de_batch_norm)
    output = decoder(z)
    activation = get_activation(de_activation)
    output = activation(output)
    bn_size = output.size(1)

    if de_batch_norm_last:
        bn = nn.BatchNorm1d(bn_size)
        output = bn(output)

    ln = nn.Linear(bn_size, npoints * 3)
    output = ln(output)
    output = output.reshape(-1, npoints, 3)
    return output
'''


if __name__=='__main__':
    input_ori = torch.randn(5, 10, 3)

    input = torch.transpose(input_ori, 1, 2)        #encoder : should be this shape attri
    d_logit, d_prob = mlp_discriminator(input, en_activation='relu', en_batch_norm=True,
                      de_activation='relu', de_batch_norm=False, de_batch_norm_last=False)
    print(d_logit, d_prob)          #d_logit: 5 * 1, d_prob: 5 * 1

    mu = 0
    sigma = 0.2
    batch_size = 5
    ndims = 128
    z = Variable(torch.Tensor(np.random.normal(mu, sigma, (batch_size, ndims))))        #z: 5 * 128
    print(z.shape)
    generator_out = point_cloud_generator(z, npoints=10, de_filters=[64, 128, 512, 1024],
                          de_activation='relu', de_batch_norm=False,
                          de_batch_norm_last=False)
    print(generator_out.shape)      #the size of generator_out should like input pc: 5 * 10 * 3

    generator_out_transpose = torch.transpose(generator_out, 1, 2)
    synthetic_logit, synthetic_prob = mlp_discriminator(generator_out_transpose, en_activation='relu', en_batch_norm=True,
                      de_activation='relu', de_batch_norm=False, de_batch_norm_last=False)
    print(synthetic_logit, synthetic_prob)          #s_logit: 5 * 1, s_prob: 5 * 1


