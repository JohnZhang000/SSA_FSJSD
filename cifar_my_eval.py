# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
from random import random
import shutil
import time

import augmentations
from models.allconv import AllConvNet
import numpy as np
from models.third_party.ResNeXt_DenseNet.models.densenet import densenet
from models.third_party.ResNeXt_DenseNet.models.resnext import resnext29
from models.third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import logging
import general as g
import socket
from scipy.fftpack import dct,idct
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',now)
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='allconv',
    choices=['wrn', 'allconv', 'densenet', 'resnext'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 10]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default=saved_dir,
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='./snapshots_std/checkpoint.pth.tar',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=16,
    help='Number of pre-fetching threads.')
parser.add_argument(
    '--imp_thresh',
    type=float,
    default=0.5,
    help='r thresh for impulse noise')
parser.add_argument(
    '--contrast_scale',
    type=float,
    default=1.0,
    help='r thresh for impulse noise')
parser.add_argument(
    '--noise_scale',
    type=float,
    default=0.5,
    help='r thresh for impulse noise')
parser.add_argument(
    '--topk',
    type=float,
    default=0.5,
    help='r thresh for impulse noise')
parser.add_argument(
    '--topk_epoch',
    type=float,
    default=0.6,
    help='r thresh for impulse noise')
parser.add_argument(
    '--no_fsim',
    '-nfs',
    action='store_true',
    help='Turn off feature similiarity loss.')
parser.add_argument(
    '--no_topk',
    '-ntk',
    action='store_true',
    help='Turn off topk loss.')
parser.add_argument(
    '--no_timei',
    '-nti',
    action='store_true',
    help='Turn off time invariant loss.')
parser.add_argument(
    '--alpha',
    type=float,
    default=3.0,
    help='r thresh for impulse noise')

args = parser.parse_args()
augmentations.IMPULSE_THRESH = args.imp_thresh
augmentations.CONTRAST_SCALE = args.contrast_scale
augmentations.NOISE_SCALE = args.noise_scale
TOPK=args.topk
TOPk_EPOCH=int(args.topk_epoch*args.epochs)

CORRUPTIONS = [
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
    'gaussian_noise', 'impulse_noise',  'shot_noise', 
    'brightness', 'fog', 'frost','snow',
    'gaussian_blur', 'saturate', 'spatter', 'speckle_noise'
]


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations_std
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  depths=[]
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 10)
    depths.append(float(depth))
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  hypers=list(ws)+[m]+depths
  return mixed,np.array(hypers)


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      aug1,depth1=aug(x, self.preprocess)
      aug2,depth2=aug(x, self.preprocess)
      im_tuple = (self.preprocess(x), aug1,
                  aug2)
      return im_tuple, y, (depth1,depth2)

  def __len__(self):
    return len(self.dataset)

class AugMixTestset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      aug1,hypers1=aug(x, self.preprocess)
      im_tuple = (self.preprocess(x), aug1)
      return im_tuple, y, hypers1

  def __len__(self):
    return len(self.dataset)

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def img2dct(clean_imgs):
    assert(4==len(clean_imgs.shape))
    assert(clean_imgs.shape[2]==clean_imgs.shape[3])
    clean_imgs=clean_imgs.transpose(0,2,3,1)
    n = clean_imgs.shape[0]
    # h = clean_imgs.shape[1]
    # w = clean_imgs.shape[2]
    c = clean_imgs.shape[3]
    
    block_dct=np.zeros_like(clean_imgs)
    sign_dct=np.zeros_like(clean_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=clean_imgs[i,:,:,j]                   
            ch_block_cln=dct2(ch_block_cln)
            sign_dct[i,:,:,j]=np.sign(ch_block_cln)
            block_dct[i,:,:,j]=np.log(1+np.abs(ch_block_cln))
    block_dct=block_dct.transpose(0,3,1,2)
    sign_dct=sign_dct.transpose(0,3,1,2)
    return block_dct,sign_dct

def dct2img(dct_imgs,signs):
    assert(4==len(dct_imgs.shape))
    assert(dct_imgs.shape[2]==dct_imgs.shape[3])
    dct_imgs=dct_imgs*signs
    dct_imgs=dct_imgs.transpose(0,2,3,1)
    n = dct_imgs.shape[0]
    c = dct_imgs.shape[3]

    images=np.zeros_like(dct_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=dct_imgs[i,:,:,j]-1
            ch_block_cln=np.exp(ch_block_cln)
            images[i,:,:,j]=ch_block_cln
    images=images.transpose(0,3,1,2)
    return images

# def select_topk(p_y,y):
#     target_onehot=torch.zeros_like(p_y).scatter_(1, y.reshape(-1,1), 1)
#     target_logits=torch.sum(p_y*target_onehot,axis=1)
#     n_correct=torch.sum((target_logits>0.5))
#     topk_aug1=target_logits.topk(int(TOPK*(len(y)-n_correct)+n_correct),dim=0)[0][-1]
#     topk=target_logits>topk_aug1
#     return topk

# def train(net, train_loader, optimizer, scheduler):
#   """Train for one epoch."""
#   net.train()
#   loss_ema = 0.
#   for i, (images, targets, depths) in enumerate(train_loader):
#     optimizer.zero_grad()

#     logits_clean,features_clean=get_preds_and_features(net,images[0].cuda(),args.model)
#     logits_aug1,features_aug1=get_preds_and_features(net,images[1].cuda(),args.model)
#     logits_aug2,features_aug2=get_preds_and_features(net,images[2].cuda(),args.model)
    
#     targets=targets.cuda()
#     # loss for pred
#     n_img=len(targets)
#     loss_pred = F.cross_entropy(logits_clean, targets)

#     # loss for jsd
#     loss_jsd=0.
#     p_clean, p_aug1, p_aug2 = F.softmax(
#         logits_clean, dim=1), F.softmax(
#             logits_aug1, dim=1), F.softmax(
#                 logits_aug2, dim=1)
#     p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

#     # features_clean=features_clean.reshape(n_img,-1)
#     # features_aug1=features_aug1.reshape(n_img,-1)
#     # features_aug2=features_aug2.reshape(n_img,-1)

#     # features_clean_mean=features_clean#torch.mean(features_clean,dim=0).reshape(1,-1).repeat_interleave(n_img,dim=0)
#     # sim_aug1=F.cosine_similarity(features_clean_mean,features_aug1,axis=-1).cuda()
#     # sim_aug2=F.cosine_similarity(features_clean_mean,features_aug2,axis=-1).cuda()
#     # scale_aug1=torch.exp(args.alpha*(1-sim_aug1))
#     # scale_aug2=torch.exp(args.alpha*(1-sim_aug2))

#     # kl_div_aug1=scale_aug1*F.kl_div(p_mixture, p_aug1, reduction='none').mean(axis=-1)
#     # kl_div_aug2=scale_aug2*F.kl_div(p_mixture, p_aug2, reduction='none').mean(axis=-1)

#     kl_div_aug1=F.kl_div(p_mixture, p_aug1, reduction='none').mean(axis=-1)
#     kl_div_aug2=F.kl_div(p_mixture, p_aug2, reduction='none').mean(axis=-1)

#     kl_div_aug1=(1+depths[0].cuda())*kl_div_aug1
#     kl_div_aug2=(1+depths[1].cuda())*kl_div_aug2

#     # topk=0.8
#     # kl_div_aug1=kl_div_aug1.topk(int(topk*n_img),dim=0)[0]
#     # kl_div_aug2=kl_div_aug2.topk(int(topk*n_img),dim=0)[0]

#     loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
#                   torch.mean(kl_div_aug1) +
#                   torch.mean(kl_div_aug2)) / 3.

#     loss=loss_pred+loss_jsd

#     loss.backward()
#     lr=optimizer.param_groups[0]['lr']
#     optimizer.step()
#     scheduler.step()
#     loss_ema = loss_ema * 0.9 + float(loss) * 0.1
#     if i % args.print_freq == 0:
#       writer.add_scalar('Lr',lr,epoch*len(train_loader)+i)
#       writer.add_scalar('Loss/Train Loss',loss_ema,epoch*len(train_loader)+i)
#       writer.add_scalar('Loss/Loss_sum',loss,epoch*len(train_loader)+i)
#       writer.add_scalar('Loss/Loss_pred',loss_pred,epoch*len(train_loader)+i)
#       writer.add_scalar('Loss/Loss_jsd',loss_jsd,epoch*len(train_loader)+i)
#       # writer.add_scalar('Scale/aug1_mean',scale_aug1.mean().detach().cpu().numpy(),epoch*len(train_loader)+i)
#       # writer.add_scalar('Scale/aug1_std',scale_aug1.std().detach().cpu().numpy(),epoch*len(train_loader)+i)
#       # writer.add_scalar('Scale/aug2_mean',scale_aug2.mean().detach().cpu().numpy(),epoch*len(train_loader)+i)
#       # writer.add_scalar('Scale/aug2_std',scale_aug2.std().detach().cpu().numpy(),epoch*len(train_loader)+i)

#       # for tag, value in net.named_parameters():
#       #   tag = tag.replace('.', '/')
#       #   writer.add_histogram(tag, value.data.cpu().numpy(), epoch*len(train_loader)+i)
#       #   writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch*len(train_loader)+i)
#   return loss_ema


# def test(net, test_loader):
#   """Evaluate network on given dataset."""
#   net.eval()
#   total_loss = 0.
#   total_correct = 0
#   with torch.no_grad():
#     for images, targets in test_loader:
#       images, targets = images.cuda(), targets.cuda()
#       logits = net(images)
#       loss = F.cross_entropy(logits, targets)
#       pred = logits.data.max(1)[1]
#       total_loss += float(loss.data)
#       total_correct += pred.eq(targets.data).sum().item()

#   return total_loss / len(test_loader.dataset), total_correct / len(
#       test_loader.dataset)


# def test_my(net, test_loader):
#   """Evaluate network on given dataset."""
#   net.eval()
#   total_loss = 0.
#   total_correct1 = 0
#   total_correct5 = 0
#   with torch.no_grad():
#     for images, targets in test_loader:
#       images, targets = images.cuda(), targets.cuda()
#       logits = net(images)
#       loss = F.cross_entropy(logits, targets)
#       acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
#       total_correct1 += float(acc1.detach().cpu().numpy())*targets.size(0)
#       total_correct5 += float(acc5.detach().cpu().numpy())*targets.size(0)

#   return total_correct1/len(test_loader.dataset), total_correct5/len(test_loader.dataset)

def test_pair(net, test_loader):
  """Evaluate network on given dataset."""
  logits_cln=[]
  logits_aug=[]
  hypers=[]
  net.eval()
  correct_cln = 0
  correct_aug = 0
  with torch.no_grad():
    for images, targets, hyper in test_loader:
      images_cln = images[0].cuda()
      images_aug = images[1].cuda()
      targets = targets.cuda()
      logit_cln = net(images_cln)
      logit_aug = net(images_aug)

      acc_cln, = accuracy(logit_cln, targets)
      acc_aug, = accuracy(logit_aug, targets)
      correct_cln += float(acc_cln.detach().cpu().numpy())*targets.shape[0]
      correct_aug += float(acc_aug.detach().cpu().numpy())*targets.shape[0]

      logit_cln=logit_cln.detach().cpu().numpy()
      logit_aug=logit_aug.detach().cpu().numpy()
      logits_cln+=[logit_cln[i,targets[i]] for i in range(len(targets))]
      logits_aug+=[logit_aug[i,targets[i]] for i in range(len(targets))]
      hypers.append(hyper)

  logits_cln=np.vstack(logits_cln)
  logits_aug=np.vstack(logits_aug)
  hypers=np.vstack(hypers)

  np.save(os.path.join(saved_dir,'logits_cln.npy'),logits_cln)
  np.save(os.path.join(saved_dir,'logits_aug.npy'),logits_aug)
  np.save(os.path.join(saved_dir,'hypers.npy'),hypers)
  return correct_cln/len(test_loader.dataset), correct_aug/len(test_loader.dataset)

def get_fourier_base(images,signs,position):
  noise=np.zeros_like(images)
  noise[:,position[0],position[1],position[2]]=1
  noise=dct2img(noise,signs)
  noise=noise/np.linalg.norm(noise,ord=2,axis=(-1,-2),keepdims=True)
  # noise=noise*4
  return noise

def add_noise(images,position):
  images=images*0.5+0.5
  images_ycbcr=g.rgb_to_ycbcr(images)
  images_dct,signs=img2dct(images_ycbcr)
  noise_ycbcr=get_fourier_base(images_dct,signs,position)

  adds=np.random.randn(images.shape[0])
  adds[adds<0.5]=-1
  adds[adds>=0.5]=1

  images_ret=images_ycbcr+adds[:,None,None,None]*noise_ycbcr
  images_ret=g.ycbcr_to_rgb(images_ret.transpose(0,2,3,1))
  images_ret=np.clip(images_ret,0,1)
  images_ret=(images_ret.transpose(0,3,1,2)-0.5)/0.5
  images_ret=images_ret.astype(np.float32)
  return images_ret

def test_single(net, test_data,position):
  """Evaluate network on given dataset."""

  test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=False)

  net.eval()
  correct=0
  with torch.no_grad():
    for images, targets in test_loader:
      images_noise = add_noise(images.numpy(),position)
      images_noise = torch.from_numpy(images_noise)
      a=images_noise-images
      print('Diff:{}'.format(a.mean()))
      images = images_noise.cuda()
      targets = targets.cuda()
      logit = net(images)

      acc, = accuracy(logit, targets)
      correct += float(acc.detach().cpu().numpy())*targets.shape[0]

  return 1-correct/len(test_loader.dataset)/100.0

def test_heatmap(net,test_data,map_size):
  heatmap=np.ones(map_size)

  with tqdm(total=np.array(map_size).prod()) as pbar:
    for i in range(map_size[0]):
      for j in range(map_size[1]):
        for k in range(map_size[2]):
          err=test_single(net,test_data,[i,j,k])
          heatmap[i,j,k]=err
          pbar.set_description('{}/{}_{}/{}_{}/{}: {}'.format(i,map_size[0],j,map_size[1],k,map_size[2],err))
  return heatmap

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def test_c(net, test_data, base_path):
#   """Evaluate network on given corrupted dataset."""
#   corruption_acc1s = []
#   corruption_acc5s = []
#   for corruption in CORRUPTIONS:
#     # Reference to original data is mutated
#     test_data.data = np.load(base_path + corruption + '.npy')
#     test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

#     test_loader_c = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=args.eval_batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False)

#     acc1, acc5 = test_my(net, test_loader_c)
#     corruption_acc1s.append(acc1)
#     corruption_acc5s.append(acc5)
#     writer.add_scalar('corruption/'+corruption+'_acc1', acc1,epoch)
#     writer.add_scalar('corruption/'+corruption+'_acc5', acc5,epoch)
#     logger.info('{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption, acc1, acc5))
#   logger.info('Corruption 15* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s[0:15]), np.mean(corruption_acc5s[0:15])))


#   return np.mean(corruption_acc1s),np.mean(corruption_acc5s)


def main():
  # torch.manual_seed(1)
  # np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)
       ])
  test_transform = preprocess

  device=socket.gethostname()
  if 'estar-403'==device: root_dataset_dir='/home/estar/Datasets'
  elif 'Jet'==device: root_dataset_dir='/mnt/sdb/zhangzhuang/Datasets'
  elif '1080x4-1'==device: root_dataset_dir='/home/zhangzhuang/Datasets'
  elif 'ubuntu204'==device: root_dataset_dir='/media/ubuntu204/F/Dataset'
  else: raise Exception('Wrong device')

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        root_dataset_dir+'/cifar-10', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        root_dataset_dir+'/cifar-10', train=False, transform=test_transform, download=True)
    base_c_path = root_dataset_dir+'/cifar-10-c/'
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
       root_dataset_dir+'/cifar-100', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
       root_dataset_dir+'/cifar-100', train=False, transform=test_transform, download=True)
    base_c_path = root_dataset_dir+'/cifar-100-c/'
    num_classes = 100
  test_data =torch.utils.data.Subset(test_data, range(0,1000))

  # train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
  # test_data  = AugMixTestset(test_data, preprocess, args.no_jsd)
  # train_loader = torch.utils.data.DataLoader(
  #     train_data,
  #     batch_size=args.batch_size,
  #     shuffle=True,
  #     drop_last=False,
  #     num_workers=args.num_workers,
  #     pin_memory=True)

  # test_loader = torch.utils.data.DataLoader(
  #     test_data,
  #     batch_size=args.eval_batch_size,
  #     shuffle=False,
  #     drop_last=False,
  #     num_workers=args.num_workers,
  #     pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  elif args.model == 'allconv':
    net = AllConvNet(num_classes)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0
  global epoch
  epoch=0


  # if args.resume:
    # if os.path.isfile(args.resume):
  checkpoint = torch.load(args.resume)

  net.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  logger.info('Model restored from {}'.format(args.resume))

  # acc_cln,acc_aug=test_pair(net, test_loader)
  map=test_heatmap(net, test_data,[3,32,32])
  my_mat={}
  for i in range(map.shape[0]):
    my_mat[str(i)]=map[i,...]
  # logger.info('Acc@cln {:.3f} Acc@aug {:.3f}'.format(acc_cln, acc_aug))
  return

  


# def get_preds_and_features(model,images,model_name):
#     # features_tmp = {}
#     # def hook(module, input, output): 
#     #   if output.device not in features_tmp.keys():
#     #     features_tmp[output.device]=[]
#     #   features_tmp[output.device].append(output)

#     # handles=[]
#     # if 'allconv'==model_name: handles.append(model.module.features.register_forward_hook(hook))
#     # elif 'wrn'==model_name: handles.append(model.module.bn1.register_forward_hook(hook))
#     # elif 'resnext'==model_name: handles.append(model.module.avgpool.register_forward_hook(hook))
#     # elif 'densenet'==model_name: handles.append(model.module.bn1.register_forward_hook(hook))
#     # else: raise Exception('model_name not supported')


#     preds = model(images)
#     # for handle in handles:
#     #   handle.remove()

#     # gather feature to the same device
#     # features=[]
#     # device_keys=list(features_tmp.keys())
#     # for key in device_keys:
#     #     features.append(features_tmp[key][0].cuda(0))
#     # features=torch.cat(features,dim=0)
#     features=0
#     return preds,features



if __name__ == '__main__':

  g.setup_seed(1)
  '''
  初始化日志系统
  '''
  set_level=logging.INFO
  logger=logging.getLogger(name='r')
  logger.setLevel(set_level)
  formatter=logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
      datefmt='%Y-%m-%d %H:%M:%S')

  fh=logging.FileHandler(os.path.join(saved_dir,'log_train.log'))
  fh.setLevel(set_level)
  fh.setFormatter(formatter)

  ch=logging.StreamHandler()
  ch.setLevel(set_level)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)
  # g.setup_seed(0)

  logger.info(args)

  writer = SummaryWriter(saved_dir)


  main()
