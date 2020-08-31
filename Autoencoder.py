import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch3d.loss import chamfer_distance
from metrics.evaluation_metrics import distChamfer, distChamferCUDA, emd_approx

def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
        "leaky_relu": F.leaky_relu,
    }
    return getter.get(argument, "Invalid activation")

class Encoder(nn.Module):
    def __init__(self, input_dim, filters, activation='relu', batch_norm=True):
        super(Encoder, self).__init__()
        self.num_layers = len(filters)
        self.input_dim = input_dim
        self.latent_dim = filters[self.num_layers - 1]
        self.batch_norm = batch_norm
        self.conv = nn.Conv1d(self.input_dim, filters[0], 1)
        self.conv_list = nn.ModuleList([
                nn.Conv1d(filters[i - 1], filters[i], 1)  for i in range(1, self.num_layers)
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(filters[i]) for i in range(0, self.num_layers)
        ])
        self.activation = get_activation(activation)
    def forward(self, x):
        if self.batch_norm:
            for i in range(0, self.num_layers):
                if i == 0:
                    x = self.conv(x)
                else:
                    x = self.conv_list[i - 1](x)

                if self.activation == F.leaky_relu:
                    x = F.leaky_relu(self.bn_list[i](x), negative_slope=0.2)
                else:
                    x = self.activation(self.bn_list[i](x))
        else:
            for i in range(self.num_layers):
                if i == 0:
                    x = self.conv(x)
                else:
                    x = self.conv_list[i - 1](x)
                if self.activation == F.leaky_relu:
                    x = F.leaky_relu(x, negative_slope=0.2)
                else:
                    x = self.activation(x)
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_dim)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, filters, activation='relu', batch_norm=False, batch_norm_last=False):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = len(filters)
        self.batch_norm = batch_norm
        self.batch_norm_last = batch_norm_last
        self.fc = nn.Linear(self.latent_dim, filters[0])
        self.fc_layers = nn.ModuleList([
            nn.Linear(filters[i - 1], filters[i]) for i in range(1, self.num_layers)
        ])

        self.activation = get_activation(activation)
        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(filters[i]) for i in range(0, self.num_layers)
        ])
    def forward(self, x):
        for i in range(0, self.num_layers - 1):
            if i == 0:
                x = self.fc(x)
            else:
                x = self.fc_layers[i - 1](x)

            if self.batch_norm:
                x = self.activation(self.bn_list[i](x))
            else:
                x = self.activation(x)

        if self.batch_norm_last:
            x = self.activation(self.bn_list[-1](self.fc_layers[-1](x)))
        else:
            x = self.activation(self.fc_layers[-1](x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128,
                 encoder_filters=[], decoder_filters=[],
                 en_batch_norm=True, en_activation='relu',
                 de_batch_norm=False, de_batch_norm_last=False, de_activation='relu',
                 npoints=2048):
        super(Autoencoder, self).__init__()
        self.Encoder = Encoder(input_dim, encoder_filters, en_activation, en_batch_norm)
        self.Decoder = Decoder(latent_dim,decoder_filters, de_activation, de_batch_norm, de_batch_norm_last)
        self.npoints = npoints

    def forward(self, x, optimizer, step, loss_type, accelerated_cd, writer_train=None):
        if writer_train is not None:
            optimizer.zero_grad()

        gen = self.Encoder(x)
        gen = self.Decoder(gen)
        gen = gen.reshape(-1, 3, self.npoints)
        gen = torch.transpose(gen, 1, 2)

        x = torch.transpose(x, 1, 2)
        loss, cd, emd = get_loss(gen, x, loss_type, accelerated_cd)

        if writer_train is not None:
            loss.backward()
            optimizer.step()
            writer_train.add_scalar('loss', loss, step)

        return gen, loss, cd, emd


def get_loss(generated, target, loss_type, accelerated_cd):
    #CD, _ = chamfer_distance(generated, target)
    generated = generated.contiguous()
    target = target.contiguous()
    if accelerated_cd:
        cdl, cdr = distChamferCUDA(generated, target)
    else:
        cdl, cdr = distChamfer(generated, target)
    #cd = cdl.mean() + cdr.mean()
    cd = torch.mean(cdl) + torch.mean(cdr)
    #print("pytorch CD: ", CD)
    print("cd: ", cd)

    emd = emd_approx(generated, target)
    emd = torch.mean(emd)
    print("emd ", emd)

    if loss_type == 'cd':
        loss = cd
    elif loss_type == 'emd':
        loss = emd

    #return loss, CD, emd
    return loss, cd, emd


if __name__=="__main__":
    input = torch.randn(5, 3, 10)           #输入必须得是这种形式

    #input = input.reshape(5, 3, 10)
    encoder = Encoder(3, [64, 128, 128, 256, 128], 'relu', True)
    decoder = Decoder(128, [256, 256, 10 * 3], 'relu', False, False)

    output = encoder(input)
    print('encoder size: ', output.shape)
    output = decoder(output)
    print('decoder size: ', output.shape)

    '''
    net = Autoencoder(3, 128, [64, 128, 128, 256, 128], [256, 256, 10 * 3], npoints=10)
    gen, loss, cd, emd = net(input, optimizer="Adam", step=0, loss_type="cd", accelerated_cd=True, writer_train=None)
    print(gen.size())

    vis_input = torch.transpose(input[0], 0, 1)
    vis_output = gen[0]
    print(vis_input.size())
    print(vis_output.size())


    #loss, CD, emd = get_loss(output, input, 'cd', False)
    #print(loss, CD, emd)
    '''











