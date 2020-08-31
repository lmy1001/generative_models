
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
from gan import vanilla_gan
from generators_discriminators import mlp_discriminator, point_cloud_generator
from Autoencoder import save

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/Users/lmy/Documents/Exercises/generative_models/data',
                    #default='/srv/beegfs-benderdata/scratch/density_estimation/data/3DMultiView/ShapeNetCore.v2.PC15k/',
                    help='the data directory')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'SGD'], help='choose the optimizer')
parser.add_argument('--momentum', type=float, default=0.9,help='Momentum for SGD')
parser.add_argument('--generator_log_name', type=str, default='log/vanilla/generator_log',
                    help='File name of generator log event')
parser.add_argument('--discriminator_log_name', type=str, default='log/vanilla/discriminator_log',
                    help='File name of discriminator log event')

parser.add_argument('--input_dim', type=int, default=3, help='dimension of input dim')
parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent dim')
parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
parser.add_argument('--learn_rate', type=float, default=(1e-4), help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--npoints', type=int, default=2048, help='input point size of one pc')
parser.add_argument('--z_dim', type=int, default=128, help='z dimensions of the generator')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='vanilla', help='output folder')
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

args = parser.parse_args()

args.manualSeed = np.random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# initialize datasets and loaders
#get train and val
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
net_d = mlp_discriminator(input_dim=args.input_dim, en_batch_norm=True, en_activation='leaky_relu',
                          de_batch_norm=True, de_activation='relu')
net_g = point_cloud_generator(latent_dim=args.latent_dim, npoints=args.npoints)
net_d = net_d.to(device)
net_g = net_g.to(device)

optimizer_d = optim.Adam(list(net_d.parameters()), lr=args.learn_rate,  betas=(0.5, 0.999))
optimizer_g = optim.Adam(list(net_g.parameters()), lr=args.learn_rate, betas=(0.5, 0.999))

'''
net=vanilla_gan(generator=net_g, discriminator=net_d)
net = net.to(device)
'''
if args.model !='':
    ckpt_d = torch.load('./vanilla/vanilla_d.pth')
    net_d.load_state_dict(ckpt_d['model'], strict=False)
    start_epoch = ckpt_d['epoch']

    ckpt_g = torch.load('./vanilla/vanilla_g.pth')
    net_g.load_state_dict(ckpt_g['model'], strict=False)
else:
    start_epoch = 0

# output log to writer
writer_g = SummaryWriter(logdir=args.generator_log_name)
writer_d = SummaryWriter(logdir=args.discriminator_log_name)

vis = visdom.Visdom()

for epoch in range(start_epoch, args.nepoch):
    epoch_loss_d, epoch_loss_g = 0, 0

    #mu, sigma = 0, 1

    for i, data in enumerate(train_loader, 0):
        net_d.train()
        net_g.train()

        step = epoch * num_batch + i

        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], \
                                        data['test_points']

        input = tr_batch.to(device)
        print("input.size: ", input.size())
        print(input[i].size())
        x_transpose = torch.transpose(input, 1, 2)
        real_logit, real_prob = net_d(x_transpose)

        z = torch.randn(input.size(0), args.z_dim)
        z = z.to(device)
        gen_out = net_g(z)
        #gen_out_transpose = torch.transpose(gen_out, 1, 2)
        synthetic_logit, synthetic_prob = net_d(gen_out)

        loss_d = torch.mean(-torch.log(real_prob) - torch.log(1 - synthetic_prob))

        loss_d.backward()
        optimizer_d.step()
        epoch_loss_d += loss_d


        for params in net_d.parameters():
            params.requires_grad = False


        z = torch.randn(input.size(0), args.z_dim)
        z = z.to(device)
        gen_out = net_g(z)
        #gen_out_transpose = torch.transpose(gen_out, 1, 2)
        synthetic_logit, synthetic_prob = net_d(gen_out)
        loss_g = torch.mean(-torch.log(synthetic_prob))


        #gen_out, _, loss_g = net(input, z)
        loss_g.backward()
        optimizer_g.step()

        #print("[epoch/%d, batch/%d]: [loss_d] %f [loss_g] %f" % (epoch, i, loss_d.item(), loss_g.item()))

        epoch_loss_g += loss_g

        if writer_d is not None and writer_g is not None:
            writer_d.add_scalar('loss', loss_d.item(), step)
            writer_g.add_scalar('loss', loss_g.item(), step)

    epoch_loss_d /= float(num_batch)
    epoch_loss_g /= float(num_batch)

    print("[epoch/%d]: [epoch_loss_d] %f [epoch_loss_g] %f" % (epoch, epoch_loss_d.item(), epoch_loss_g.item()))


save(net_d, optimizer_d, epoch + 1, '%s/vanilla_d.pth' % (args.outf))
save(net_g, optimizer_g, epoch + 1, '%s/vanilla_g.pth' % (args.outf))

writer_d.close()
writer_g.close()

for i, data in enumerate(test_loader, 0):
    #net.eval()
    net_d.eval()
    net_g.eval()
    mu, sigma = 0, 0.2
    if i % 10 == 0:
        with torch.no_grad():
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], \
                                    data['test_points']
            input = tr_batch.to(device)
            x_transpose = torch.transpose(input, 1, 2)
            real_logit, real_prob = net_d(x_transpose)
        
            z_tmp = np.random.normal(mu, sigma, (input.size(0), args.z_dim))
            z_tmp = z_tmp.astype(np.float32)
            z = torch.from_numpy(z_tmp)
            z = z.to(device)
            gen_out = net_g(z)

            gen_out_transpose = torch.transpose(gen_out, 1, 2)
            synthetic_logit, synthetic_prob = net_d(gen_out_transpose)

            loss_d = torch.mean(-torch.log(real_prob) - torch.log(1 - synthetic_prob))
            loss_g = torch.mean(-torch.log(synthetic_prob))
            
            #gen_out, loss_d, loss_g = net(input, z)
            print("[batch %d]: loss_d %f  loss_g %f", i, loss_d.item(), loss_g.item())

