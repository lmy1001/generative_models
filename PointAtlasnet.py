import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
#from pytorch3d.loss import chamfer_distance
#import visdom
from Autoencoder import get_loss, get_activation

class Pointnet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Pointnet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim)
        self.bn5 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_dim)         #size: N * latent_dim
        x = F.relu(self.bn4(self.fc1(x).unsqueeze(-1)))            #unsqueeze: N * latent_dim * 1
        x = F.relu(self.bn5(self.fc2(x.squeeze(2)).unsqueeze(-1)))     #squeeze: N * latent_dim, unsqueeze: N * latent_dim * 1
        return x.squeeze(2)


class Atlasnet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3,
                 bottleneck_size=1024, num_layers=2,
                 hidden_neurons=512, activation='relu'):
        super(Atlasnet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_size = bottleneck_size
        self.hidden_neurons = hidden_neurons
        self.num_layers = num_layers
        self.activation = activation
        self.conv1 = nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
        self.conv2 = nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList(
            [nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])
        self.last_conv = nn.Conv1d(self.hidden_neurons, self.output_dim, 1)

        self.bn1 = nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons)
        self.bn_list = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(self.activation)

    def forward(self, x, latent_vec):
        x = self.conv1(x) + latent_vec          # N * latent_dim * 1
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        x = self.last_conv(x)

        return x


class PointAtlasnet(nn.Module):

    def __init__(self, input_dim=3, output_dim=3,
                 bottleneck_size=1024, num_layers=2,
                 hidden_neurons=512, activation='relu'):
        super(PointAtlasnet, self).__init__()
        self.encoder = Pointnet(input_dim=input_dim, latent_dim=bottleneck_size)
        self.decoder = Atlasnet(input_dim=input_dim,
                                output_dim=output_dim,
                                bottleneck_size=bottleneck_size,
                                num_layers=num_layers,
                                hidden_neurons=hidden_neurons,
                                activation=activation)

    def forward(self, x, step, optimizer, loss_type, accelerated_cd, writer_train):

        if writer_train is not None:
            optimizer.zero_grad()

        latent_vec = self.encoder(x)                #input is N * 3 * 2048
        # decoder_input = x.unsqueeze(1)              #decoder input should be : N * 1 * 3 * 2048
        # latent_vec = latent_vec.unsqueeze(2).unsqueeze(1)       #latent_vec: N * 1 * latent_dim * 1
        latent_vec = latent_vec.unsqueeze(2)  # size: N * latent_dim * 1
        output = self.decoder(x, latent_vec)        #decoder: N * 3 * 2048

        output = torch.transpose(output, 1, 2)      # N * 2048 * 3

        x = torch.transpose(x, 1, 2)                # N * 2048 * 3

        loss, cd, emd = get_loss(output, x, loss_type, accelerated_cd)

        if writer_train is not None:
            loss.backward()
            optimizer.step()

            writer_train.add_scalar('loss', loss.item(), step)

        return output, loss, cd, emd

    def get_latent_codes(self, x):
        latent_codes = self.encoder(x)
        return latent_codes



if __name__=='__main__':
    x = Variable(torch.randn(100, 3, 2048))
    net = PointAtlasnet()           #Pointnet size: N * latent_dim
    out = net(x)
    print(out.shape)
    x = torch.transpose(x, 1, 2)
    print(x.shape)
    out = torch.transpose(out, 1, 2)
    print(out.shape)
    #loss_chamfer, _ = chamfer_distance(out, x)
    #print(loss_chamfer)

    vis = visdom.Visdom()
    x_fig = vis.scatter(x[0])
    out_fig = vis.scatter(out[0])