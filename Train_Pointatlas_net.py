import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import argparse
import datasets
import os
from Autoencoder import get_loss, save
from PointAtlasnet import PointAtlasnet
from tensorboardX import SummaryWriter
import visdom
from metrics.evaluation_metrics import compute_all_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/Users/lmy/Documents/Exercises/generative_models/data',
                    #default='/srv/beegfs-benderdata/scratch/density_estimation/data/3DMultiView/ShapeNetCore.v2.PC15k/',
                    help='the data directory')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'SGD'], help='choose the optimizer')
parser.add_argument('--momentum', type=float, default=0.9,help='Momentum for SGD')
parser.add_argument('--train_log_name', type=str, default='log/pointatlas/train_log',
                    help='File name of train log event')
parser.add_argument('--val_log_name', type=str, default='log/pointatlas/val_log',
                    help='File name of validation log event')

parser.add_argument('--input_dim', type=int, default=3, help='dimension of input dim')
parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent dim')
parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
parser.add_argument('--learn_rate', type=float, default=(5*1e-4), help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
parser.add_argument('--npoints', type=int, default=2048, help='input point size of one pc')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='pointatlas', help='output folder')
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

net = PointAtlasnet(input_dim=args.input_dim, output_dim=args.input_dim)

if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(net.parameters()), lr=args.learn_rate,  betas=(0.9, 0.999))
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(net.parameters()), lr=args.learn_rate, momentum=args.momentum)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if args.model !='':
    ckpt = torch.load(args.model)
    net.load_state_dict(ckpt['model'], strict=True)
    start_epoch = ckpt['epoch']
else:
    start_epoch = 0

#device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
net = net.cuda()

# output log to writer
writer_train = SummaryWriter(logdir=args.train_log_name)
writer_val = SummaryWriter(logdir=args.val_log_name)

vis = visdom.Visdom()
for epoch in range(start_epoch, args.nepoch):

    for i, data in enumerate(train_loader, 0):
        step = i + len(train_loader) * epoch
        idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
        net.train()

        input = tr_batch.cuda()
        input = torch.transpose(input, 1, 2)
        output, loss, cd, emd = net(input, step, optimizer, args.loss_type, args.accelerated_cd, writer_train)
        output = output.cuda()

        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            vis_input = torch.transpose(input[i], 0, 1)
            vis_output = output[i]
            vis.scatter(vis_input)
            vis.scatter(vis_output)
            j, data = next(enumerate(val_loader, 0))
            j_batch, tr_val_batch, te_val_batch = data['idx'], data['train_points'], data['test_points']
            net.eval()
            with torch.no_grad():
                val_input = tr_val_batch.cuda()
                val_input = torch.transpose(val_input, 1, 2)
                output, val_loss, cd, emd = net(val_input, step, optimizer, args.loss_type, args.accelerated_cd, writer_train=None)
                output = output.cuda()
                print('[%d: %d/%d] val loss: %f' % (epoch, i, val_num_batch, val_loss.item()))

            if writer_val is not None:
                writer_val.add_scalar('loss', val_loss.item(), step)
save(net, optimizer, epoch + 1,  '%s/pointatlas_last_model_%d.pth' % (args.outf, epoch))

writer_train.close()
writer_val.close()

# test with mmd_cd and mmd_emd
all_gen_pc = []
all_input_pc = []
num_all_pcs = 0
for i, data in enumerate(test_loader, 0):
    step = i
    idx_batch, tr_test_batch, te_test_batch = data['idx'], data['train_points'], data['test_points']
    net.eval()
    with torch.no_grad():
        test_input = tr_test_batch.cuda()
        test_input = torch.transpose(test_input, 1, 2)
        test_output, test_loss, cd, emd = net(test_input, step, optimizer, args.loss_type, args.accelerated_cd, writer_train=None)
        test_output = test_output.cuda()

        all_gen_pc.append(test_output)
        all_input_pc.append(tr_test_batch.cuda())

        num_all_pcs += test_input.size(0)

sample_pcs = torch.cat(all_gen_pc, dim=0)
ref_pcs = torch.cat(all_input_pc, dim=0)
print("Generation Sample size:%s Ref size: %s"
        % (sample_pcs.size(), ref_pcs.size()))

#calculate mmd_cd and mmd_emd
res = compute_all_metrics(sample_pcs,ref_pcs, args.batch_size, args.accelerated_cd)
print(res)







