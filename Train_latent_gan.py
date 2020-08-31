import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import argparse
import datasets
import os
from tensorboardX import SummaryWriter
import visdom
from gan import vanilla_gan, w_gan_gp
from generators_discriminators import latent_code_generator_two_layers, \
                                    latent_code_discriminator_twp_layers
from PointAtlasnet import PointAtlasnet

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    #default='/Users/lmy/Documents/Exercises/generative_models/data',
                    default='/srv/beegfs-benderdata/scratch/density_estimation/data/3DMultiView/ShapeNetCore.v2.PC15k/',
                    help='the data directory')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'SGD'], help='choose the optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--generator_log_name', type=str, default='log/latent_gan/generator_log',
                    help='File name of generator log event')
parser.add_argument('--discriminator_log_name', type=str, default='log/latent_gan/discriminator_log',
                    help='File name of discriminator log event')

parser.add_argument('--input_dim', type=int, default=3, help='dimension of input dim')
parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent dim')
parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
parser.add_argument('--learn_rate', type=float, default=(1e-4), help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--npoints', type=int, default=2048, help='input point size of one pc')
parser.add_argument('--z_dim', type=int, default=128, help='z dimensions of the generator')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='latent_gan', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                    help='Max number of sampled points (train)')
parser.add_argument("--te_max_sample_points", type=int, default=2048,
                    help='Max number of sampled points (test)')
parser.add_argument('--normalize_per_shape', action='store_true',
                    help='Whether to perform normalization per shape.')
parser.add_argument('--normalize_std_per_axis', action='store_true',
                    help='Whether to perform normalization per axis.')
parser.add_argument('--dataset_scale', type=float, default=1.,
                    help='Scale of the dataset (x,y,z * scale = real output, default=1).')
parser.add_argument('--cates', type=str, nargs='+', default=["chair"],
                    help="Categories to be trained (useful only if 'shapenet' is selected)")
parser.add_argument('---loss_type', type=str, default='cd',
                    help='decide the type of loss during training')
parser.add_argument('--accelerated_cd', type=bool, default=True,
                    help='decide whether to use cuda version of cd or just cd calculation')

parser.add_argument('gan_type', type=str, default='vanilla',
                    help='choose the gan type to use')

args = parser.parse_args()

# initialize datasets and loaders
# get train and val
tr_dataset = datasets.ShapeNet15kPointClouds(
    categories=args.cates, split='train',
    tr_sample_size=args.tr_max_sample_points,
    te_sample_size=args.te_max_sample_points,
    scale=args.dataset_scale, root_dir=args.data_dir,
    normalize_per_shape=args.normalize_per_shape,
    normalize_std_per_axis=args.normalize_std_per_axis,
    random_subsample=True)

val_dataset = datasets.ShapeNet15kPointClouds(
    categories=args.cates, split='val',
    tr_sample_size=args.tr_max_sample_points,
    te_sample_size=args.te_max_sample_points,
    scale=args.dataset_scale, root_dir=args.data_dir,
    normalize_per_shape=args.normalize_per_shape,
    normalize_std_per_axis=args.normalize_std_per_axis,
    all_points_mean=tr_dataset.all_points_mean,
    all_points_std=tr_dataset.all_points_std,
)

te_dataset = datasets.ShapeNet15kPointClouds(
    categories=args.cates, split='test',
    tr_sample_size=args.tr_max_sample_points,
    te_sample_size=args.te_max_sample_points,
    scale=args.dataset_scale, root_dir=args.data_dir,
    normalize_per_shape=args.normalize_per_shape,
    normalize_std_per_axis=args.normalize_std_per_axis,
    all_points_mean=tr_dataset.all_points_mean,
    all_points_std=tr_dataset.all_points_std,
)

train_loader = torch.utils.data.DataLoader(
    dataset=tr_dataset, batch_size=args.batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    dataset=te_dataset, batch_size=args.batch_size, shuffle=False)
print(len(train_loader), len(val_loader), len(test_loader))

num_batch = len(train_loader)

val_num_batch = len(val_loader)
test_num_batch = len(test_loader)

try:
    os.makedirs(args.outf)
except OSError:
    pass

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
PointAtlasnet = PointAtlasnet(input_dim=3, output_dim=3,
                 bottleneck_size=1024, num_layers=2,
                 hidden_neurons=512, activation='relu')
pt = torch.load()
PointAtlasnet.load_state_dict(pt['model'], strict=True)

net_d = latent_code_discriminator_twp_layers(args.latent_dim)
net_g = latent_code_generator_two_layers(args.latent_dim)

net_d = net_d.to(device)
net_g = net_g.to(device)

optimizer_d = optim.Adam(list(net_d.parameters()), lr=args.learn_rate, betas=(0.5, 0.999))
optimizer_g = optim.Adam(list(net_g.parameters()), lr=args.learn_rate, betas=(0.5, 0.999))

if args.net_type == 'vanilla_gan':
    net = vanilla_gan(generator=net_g, discriminator=net_d)
else:
    net = w_gan_gp(generator=net_g, discriminator=net_d)
net = net.to(device)

if args.model != '':
    ckpt = torch.load(args.model)
    net.load_state_dict(ckpt['model'], strict=True)
    start_epoch = ckpt['epoch']
else:
    start_epoch = 0

# output log to writer
writer_g = SummaryWriter(logdir=args.generator_log_name)
writer_d = SummaryWriter(logdir=args.discriminator_log_name)

vis = visdom.Visdom()

for epoch in range(start_epoch, args.nepoch):
    epoch_loss_d, epoch_loss_g = 0, 0
    discriminator_boost = 5
    iterations_for_epoch = int(num_batch / discriminator_boost)
    mu, sigma = 0, 0.2

    for i, data in enumerate(train_loader, 0):
        net.train()

        optimizer_d.zero_grad()
        idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], \
                                        data['test_points']

        input = tr_batch.to(device)

        latent_codes = PointAtlasnet.get_latent_codes(input)
        latent_codes = latent_codes.to(device)
        gen_out, loss_d, loss_g = net(input, latent_codes)
        loss_d.backward(retain_graph=True)
        optimizer_d.step()
        epoch_loss_d += loss_d

        for params in net_d.parameters():
            params.requires_grad = False

        if args.gan_type == 'vanilla':
            loss_g.backward()
            optimizer_g.step()
            epoch_loss_g += loss_g
        else:
            if i % discriminator_boost == 1:
                loss_g.backward()
                optimizer_g.step()
                epoch_loss_g += loss_g

    epoch_loss_d /= float(num_batch)
    if args.gan_type == 'vanilla':
        epoch_loss_g /= float(num_batch)
    else:
        epoch_loss_g /= float(iterations_for_epoch)

    print('[epoch %d]: [epoch_loss_d] %f [epoch_loss_g] %f', epoch, epoch_loss_d, epoch_loss_g)
    if writer_d is not None and writer_g is not None:
        writer_d.add_scalar('loss_d', epoch_loss_d, epoch)
        writer_g.add_scalar('loss_g', epoch_loss_g, epoch)

torch.save(net, epoch + 1,  '%s/cls_last_model_%d.pth' % (args.outf, epoch))
for i, data in enumerate(test_loader, 0):
    net.eval()
    if i % 10 == 0:
        with torch.no_grad():
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], \
                                            data['test_points']
            input = tr_batch.to(device)

            latent_codes = PointAtlasnet.get_latent_codes(input)
            latent_codes = latent_codes.to(device)

            gen_out, loss_d, loss_g = net(input, latent_codes)
            print("[batch %d]: losdd_d %f  loss_g %f")
            vis.scatter(input[i], name='gt')
            vis.scatter(gen_out[i], name='gen_out')

