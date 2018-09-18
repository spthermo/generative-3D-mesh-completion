from __future__ import print_function
import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import dataIO as d
import visdom
from logger import Logger

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--localSize', type=int, default=32, help='the height / width of the region around the mask')
parser.add_argument('--hole_min', type=int, default=12, help='min height / width of the mask')
parser.add_argument('--hole_max', type=int, default=18, help='max height / width of the mask')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50, help='number of epochs')
parser.add_argument('--preniter', type=int, default=20, help='number of epochs for generator pretraining')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--alpha', type=float, default=0.01, help='the weight of discriminator loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam. default=0.99')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ndf = int(opt.ndf)
nc = 3
obj = 'chair'
obj_ratio = 0.7
is_local = False

#create a directory to save the logs
if not os.path.exists('logs'):
os.makedirs('logs')

# save feedback
logger = Logger('./logs')

# init visdom server for visualization
vis = visdom.Visdom()

#create a directory to save the trained model
if not os.path.exists('models'):
os.makedirs('models')


# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=8, dilation=8, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.view(-1, 1).squeeze(1)


# weight initialization
def weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


# generate "cuboid" noise
def get_points():
    points = []
    mask = []
    for i in range(opt.batchSize):
        x1, y1, z1 = np.random.randint(0, opt.imageSize - opt.localSize + 1, 3)
        x2, y2, z2 = np.array([x1, y1, z1]) + opt.localSize
        points.append([x1, y1, x2, y2, z1, z2])

        w, h, d = np.random.randint(opt.hole_min, opt.hole_max + 1, 3)
        p1 = x1 + np.random.randint(0, opt.localSize - w)
        q1 = y1 + np.random.randint(0, opt.localSize - h)
        r1 = z1 + np.random.randint(0, opt.localSize - d)
        p2 = p1 + w
        q2 = q1 + h
        r2 = r1 + d

        m = np.zeros((1, opt.imageSize, opt.imageSize, opt.imageSize), dtype=np.uint8)
        m[:, q1:q2 + 1, p1:p2 + 1, r1:r2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)


def save_checkpoint(state, curr_epoch):
    torch.save(state, './models/netG_e%d.pth.tar' % (curr_epoch))


# initialize Generator & Discriminator
netG = Generator().to(device)
weights_init(netG)
print(netG)

netD = Discriminator().to(device)
weights_init(netD)
print(netD)

# load ".off" files
volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
print('Using ' + obj + ' Data')
volumes = volumes[..., np.newaxis].astype(np.float)
data = torch.from_numpy(volumes)
data = data.permute(0, 4, 1, 2, 3)
data = data.type(torch.FloatTensor)


# choose loss function
criterion = nn.BCELoss()
criterion2 = nn.MSELoss()

# fake/real labels
real_label = 1
fake_label = 0

# setup optimizers
optG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

errD_all = AverageMeter()
errG_all = AverageMeter()

# start training
t_batches = int(data.size(0) / opt.batchSize)
for epoch in range(opt.niter):
    t0 = time.time()
    data_perm = torch.randperm(data.size(0))

    for i in range(t_batches):

        # create random "cuboid" noise
        points_batch, mask_batch = get_points()

        batch = data[i*opt.batchSize:(i*opt.batchSize + opt.batchSize)]
        real_data = batch.to(device)

        # add noise to batch
        temp = torch.from_numpy(mask_batch)
        masks = temp.type(torch.FloatTensor).cuda()
        masked_data = real_data + masks
        masked_data[masked_data > 1] = 1

        # warm up the Generator for a few epochs
        if epoch <= opt.preniter:
            optG.zero_grad()
            gen_data = netG(masked_data)
            errG = criterion2(gen_data, real_data)
            errG.backward()
            optG.step()

            print('PRETRAIN [%d/%d][%d/%d] Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, t_batches, errG.item()))

            errG_all.update(errG.item())
        # train both Generator and Discriminator
        else:
            optD.zero_grad()

            # train Discriminator with real samples
            label = torch.full((opt.batchSize,), real_label, device=device)
            out = netD(real_data)
            errD_real = criterion(out, label)

            # train Discriminator with generated samples
            label_fake = label.clone()
            label_fake.fill_(fake_label)
            gen_data = netG(masked_data)
            out = netD(gen_data.detach())
            errD_fake = criterion(out, label_fake)
            errD = (errD_real + errD_fake) * opt.alpha
            errD.backward()
            optD.step()

            # update Generator
            optG.zero_grad()
            gen_data = netG(masked_data)
            out = netD(gen_data)
            errG = criterion2(gen_data, real_data)
            errG.backward()
            optG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, t_batches, errD, errG))

            errD_all.update(errD.item())
            errG_all.update(errG.item())

        # visdom visualization
        if (i % 50) == 0 and i > 0:
            vis.close(None)
            id_ch = np.random.randint(0, opt.batchSize, opt.batchSize)
            t = gen_data.detach().cpu().clone()
            t = t.permute(0, 4, 2, 3, 1)
            gen_data_np = t.numpy()
            t = masked_data.detach().cpu().clone()
            t = t.permute(0, 4, 2, 3, 1)
            masked_data_np = t.numpy()
            t = real_data.detach().cpu().clone()
            t = t.permute(0, 4, 2, 3, 1)
            real_data_np = t.numpy()
            for j in range(opt.batchSize):
                if gen_data_np[id_ch[j]].max() > 0.5:
                    d.plotVoxelVisdom(np.squeeze(real_data_np[id_ch[j]] > 0.5), vis, '_'.join(map(str, [epoch, j])))
                    d.plotVoxelVisdom(np.squeeze(masked_data_np[id_ch[j]] > 0.5), vis, '_'.join(map(str, [epoch, j+1])))
                    d.plotVoxelVisdom(np.squeeze(gen_data_np[id_ch[j]] > 0.5), vis, '_'.join(map(str, [epoch, j+2])))
                    break

    print('Time elapsed Epoch %d: %d seconds'
            % (epoch + 1, time.time() - t0))

    # TensorBoard logging
    # scalar values
    info = {
        'D loss': errD_all.avg,
        'G loss': errG_all.avg
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # values and gradients of the parameters (histogram)
    for tag, value in netG.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), epoch)

# save Generator parameters and optimizer (last epoch)
save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': netG.state_dict(),
    'optimizer': optG.state_dict(),
}, epoch + 1)
