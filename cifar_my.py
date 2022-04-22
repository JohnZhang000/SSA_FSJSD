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
from distutils.debug import DEBUG

# import misc
import argparse
import os
# from sched import scheduler
import shutil
import time
from PIL import Image

import augmentations
# from augmentations import my_noise
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
# import albumentations
# from pytorch_metric_learning.losses import NTXentLoss
# from pytorch_metric_learning.utils import distributed as pml_dist
import socket
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
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
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
    default='',# ./snapshots/allconv_checkpoint.pth.tar #'./results/2022-04-04-23_34_11/checkpoint.pth.tar',#
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--eval-crpt',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=16,
    help='Number of pre-fetching threads.')

# distributed training parameters
# parser.add_argument('--world_size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--local_rank', default=-1, type=int)
# parser.add_argument('--dist_on_itp', action='store_true')
# parser.add_argument('--dist_url', default='env://',
#                     help='url used to set up distributed training')


args = parser.parse_args()

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

# def webpf(img):
#     img=np.array(img)
#     quality=np.random.randint(20,100)
#     assert(img.shape[0]==img.shape[1])  
#     # assert(img.max()>1)
#     webp_aug = albumentations.Compose([
#         albumentations.ImageCompression(quality_lower=quality,quality_upper=quality,compression_type=0,p=1),
#         # albumentations.HorizontalFlip(p=0.5)
#         ])
    
#     augmented = webp_aug(image=img)
#     auged = augmented['image']/255.0
#     auged = torch.from_numpy(auged).permute(2,0,1).float()
#     # img = Image.fromarray(img.astype('uint8')).convert('RGB')
#     return auged



def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


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
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

class My_AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.len=self.__len__()

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      x_neg, y_neg = self.dataset[np.random.randint(0,self.len)]
      while(y_neg==y):
        x_neg, y_neg = self.dataset[np.random.randint(0,self.len)]
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  self.preprocess(x_neg))
      y_tuple=(y,y,y_neg)
      return im_tuple, y_tuple

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    # images = images.cuda()
    # targets_pos = targets[0].cuda()
    # targets_aug = targets[1].cuda()
    # targets_neg = targets[2].cuda()
    # logits = net(images)
    logits_clean,features_clean=get_preds_and_features(net,images[0].cuda())
    logits_aug1,features_aug1=get_preds_and_features(net,images[1].cuda())
    logits_aug2,features_aug2=get_preds_and_features(net,images[2].cuda())

    # loss = F.cross_entropy(logits, targets)
    logits_all=[logits_clean, logits_aug1, logits_aug2]
    features_all=[features_clean,features_aug1,features_aug2]
    targets_all=targets.cuda()#[targets_pos,targets_aug,targets_neg]
    loss_pred,loss_jsd,loss_feature = my_loss(logits_all,features_all,targets_all)
    loss=loss_pred+loss_jsd+loss_feature

    # if args.no_jsd:
    #   images = images.cuda()
    #   targets = targets.cuda()
    #   logits = net(images)
    #   loss = F.cross_entropy(logits, targets)
    # else:
    #   images_all = torch.cat(images, 0).cuda()
    #   targets = targets.cuda()
    #   logits_all = net(images_all)
    #   logits_clean, logits_aug1, logits_aug2 = torch.split(
    #       logits_all, images[0].size(0))

    #   # Cross-entropy is only computed on clean images
    #   loss = F.cross_entropy(logits_clean, targets)

    #   p_clean, p_aug1, p_aug2 = F.softmax(
    #       logits_clean, dim=1), F.softmax(
    #           logits_aug1, dim=1), F.softmax(
    #               logits_aug2, dim=1)

    #   # Clamp mixture distribution to avoid exploding KL divergence
    #   p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    #   loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #                 F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #                 F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()
    lr=optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      # logger.info('Train Loss {:.3f}'.format(loss_ema))
      writer.add_scalar('Lr',lr,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Train Loss',loss_ema,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_sum',loss,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_pred',loss_pred,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_jsd',loss_jsd,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_feature',loss_feature,epoch*len(train_loader)+i)

      for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram(tag, value.data.cpu().numpy(), epoch*len(train_loader)+i)
        writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch*len(train_loader)+i)




  return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_my(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct1 = 0
  total_correct5 = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
      total_correct1 += float(acc1.detach().cpu().numpy())*targets.size(0)
      total_correct5 += float(acc5.detach().cpu().numpy())*targets.size(0)

  return total_correct1/len(test_loader.dataset), total_correct5/len(test_loader.dataset)


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

def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  # corruption_accs = []
  corruption_acc1s = []
  corruption_acc5s = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader_c = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False)

    # test_loss, test_acc = test(net, test_loader)
    # corruption_accs.append(test_acc)
    # print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
    # corruption, test_loss, 100 - 100. * test_acc))
    acc1, acc5 = test_my(net, test_loader_c)
    corruption_acc1s.append(acc1)
    corruption_acc5s.append(acc5)
    writer.add_scalar('corruption/'+corruption+'_acc1', acc1,epoch)
    writer.add_scalar('corruption/'+corruption+'_acc5', acc5,epoch)
    logger.info('{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption, acc1, acc5))
  logger.info('Corruption 15* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s[4:]), np.mean(corruption_acc5s[4:])))


  return np.mean(corruption_acc1s),np.mean(corruption_acc5s)
  # print('15* Mean Corruption Error@1 {:.3f}'.format(100-100*np.mean(corruption_accs[4:])))
  # return np.mean(corruption_accs)


def main():
  # misc.init_distributed_mode(args)
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
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
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar/CIFAR-100-C/'
    num_classes = 100

  train_data = AugMixDataset(train_data, preprocess, args.no_jsd)

  # if True:  # args.distributed:
  #     num_tasks = misc.get_world_size()
  #     global_rank = misc.get_rank()
  #     sampler_train = torch.utils.data.DistributedSampler(
  #         train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
  #     )
  #     sampler_test = torch.utils.data.DistributedSampler(
  #         test_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
  #     )
  #     print("Sampler_train = %s" % str(sampler_train))
  train_loader = torch.utils.data.DataLoader(
      train_data,
      # sampler=sampler_train,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=False)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      # sampler=sampler_test,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=False)

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
  # if args.distributed:
  #   net = torch.nn.parallel.DistributedDataParallel(net, device_ids=args.gpu, find_unused_parameters=True)
    # model_without_ddp = model.module

  start_epoch = 0
  global epoch
  epoch=0


  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      # start_epoch = checkpoint['epoch'] + 1
      # best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      # net.load_state_dict(checkpoint['model_state_dict'])
      # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      logger.info('Model restored from {}'.format(args.resume))

      # mce1, mce5 = test_c(net, test_data, base_c_path)
      # writer.add_scalar('corruption/corruption_mean_acc1', mce1,0)
      # writer.add_scalar('corruption/corruption_mean_acc5', mce5,0)
      # logger.info('Corruption mean * Acc@1 {:.3f} Acc@5 {:.3f}'.format(mce1, mce5))

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    # test_loss, test_acc = test(net, test_loader)
    # print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
    #     test_loss, 100 - 100. * test_acc))

    # test_c_acc = test_c(net, test_data, base_c_path)
    # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

    # acc1, acc5 = test_my(net, test_loader)
    # logger.info('Clean * Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))

    mce1, mce5 = test_c(net, test_data, base_c_path)
    # writer.add_scalar('corruption/corruption_mean_acc1', acc1)
    # writer.add_scalar('corruption/corruption_mean_acc5', mce5)

    logger.info('Corruption mean * Acc@1 {:.3f} Acc@5 {:.3f}'.format(mce1, mce5))
    return

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))
  # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*len(train_loader))
  # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  # scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,args.learning_rate,steps_per_epoch=len(train_loader), epochs=args.epochs)
  # scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer,1e-6 / args.learning_rate,args.learning_rate)



  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  logger.info('Beginning training from epoch:{}'.format(start_epoch + 1))
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()
    # train_loader.sampler.set_epoch(epoch)
    # test_loader.sampler.set_epoch(epoch)

    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    writer.add_scalar('Total/Train Loss', train_loss_ema,epoch)
    writer.add_scalar('Total/Test Loss', train_loss_ema,epoch)
    writer.add_scalar('Total/Test Acc', 100*test_acc,epoch)

    logger.info(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))
    # 此处有bug，不能打开
    # if epoch%args.eval_crpt==0:
    #   mce1, mce5 = test_c(net, test_data, base_c_path)
    #   writer.add_scalar('Total/Test_C Acc', mce1,epoch)
    #   logger.info('C_Acc1: {:.3f}, C_Acc5: {:.3f}'.format(mce1, mce5))

  mce1, mce5 = test_c(net, test_data, base_c_path)
  logger.info('C_Acc1: {:.3f}, C_Acc5: {:.3f}'.format(mce1, mce5))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - mce1))

  # test_c_acc = test_c(net, test_data, base_c_path)
  # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

  # with open(log_path, 'a') as f:
  #     f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
  #             (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))



def get_preds_and_features(model,images):
    features_tmp = {}
    def hook(module, input, output): 
      if output.device not in features_tmp.keys():
        features_tmp[output.device]=[]
      features_tmp[output.device].append(output)

    handles=[]
    handles.append(model.module.features.register_forward_hook(hook))
    # for module in model.module.features:
    #   if 'Conv2d' in module._get_name():
    #       handles.append(module.register_forward_hook(hook))

    preds = model(images)
    for handle in handles:
      handle.remove() ## hook删除 

    # 
    features=[]
    device_keys=list(features_tmp.keys())
    feature_num=len(features_tmp[device_keys[0]])
    samples_num=features_tmp[device_keys[0]][0].shape[0]
    for i in range(feature_num):
      feature_i=[]
      for key in device_keys:
          feature_i.append(features_tmp[key][i].cuda(0))
      feature_i=torch.cat(feature_i,dim=0)
      features.append(feature_i)
    # del features_tmp
    # torch.cuda.empty_cache()
    return preds,features

def get_corr(fake_Y, Y):#计算两个向量person相关系数
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
                torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

def my_loss(logits_all,features_all,targets):
    marg=0.1
    logits_clean=logits_all[0]
    logits_aug1=logits_all[1]
    logits_aug2=logits_all[2]
    
    # loss for pred
    loss_pred = F.cross_entropy(logits_clean, targets.cuda())
    # loss_pred += F.cross_entropy(logits_aug1, targets)
    # loss_pred += F.cross_entropy(logits_aug2, targets)
    # loss_pred = loss_pred/3.

    loss_feature=0.
    features_clean=features_all[0][0].reshape(len(targets),-1)
    features_aug1=features_all[1][0].reshape(len(targets),-1)
    features_aug2=features_all[2][0].reshape(len(targets),-1)
    logger.debug('features_clean: {} {} {} features_aug1: {} {} {} features_aug2: {} {} {}'.format(features_clean.min(),features_clean.max(),features_clean.mean(),
                                                                                                 features_aug1.min(),features_aug1.max(),features_aug1.mean(),
                                                                                                 features_aug2.min(),features_aug2.max(),features_aug2.mean()))

    features_clean_mean=torch.mean(features_clean,dim=0).reshape(1,-1)
    features_clean_mean=features_clean_mean.repeat_interleave(len(targets),dim=0)
    logger.debug('features_clean_mean: {} {} {}'.format(features_clean_mean.min(),features_clean_mean.max(),features_clean_mean.mean()))

    sim_aug1=F.cosine_similarity(features_clean_mean,features_aug1,axis=-1).cuda()
    logger.debug('sim_aug1: {} {} {}'.format(sim_aug1.min(),sim_aug1.max(),sim_aug1.mean()))
    sim_aug2=F.cosine_similarity(features_clean_mean,features_aug2,axis=-1).cuda()
    logger.debug('sim_aug2: {} {} {}'.format(sim_aug2.min(),sim_aug2.max(),sim_aug2.mean()))
    # sim_aug1=torch.ones(len(targets))
    # for i in range(len(sim_aug1)):
    #   sim_aug1[i]=F.cosine_similarity(features_clean_mean,features_aug1[i,...].reshape(1,-1))
    # sim_aug2=torch.ones(len(targets))
    # for i in range(len(sim_aug2)):
    #   sim_aug2[i]=F.cosine_similarity(features_clean_mean,features_aug2[i,...].reshape(1,-1))
    scale_aug1=torch.exp((sim_aug1-1)*(sim_aug1-1))#.cuda().requires_grad_(False)
    logger.debug('scale_aug1: {} {} {}'.format(scale_aug1.min(),scale_aug1.max(),scale_aug1.mean()))

    scale_aug2=torch.exp((sim_aug2-1)*(sim_aug2-1))#.cuda().requires_grad_(False)
    logger.debug('scale_aug2: {} {} {}'.format(scale_aug2.min(),scale_aug2.max(),scale_aug2.mean()))

    # scale_aug1=scale_aug1*(torch.exp(epoch-100))
    # scale_aug2=scale_aug2*(torch.exp(epoch-100))
    # loss_feature=torch.mean(torch.stack(loss_feature))

    # loss for jsd
    loss_jsd=0.
    p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    kl_div_aug1=F.kl_div(p_mixture, p_aug1, reduction='none').mean(axis=-1)
    kl_div_aug2=F.kl_div(p_mixture, p_aug2, reduction='none').mean(axis=-1)
    # for i in range(len(kl_div_aug1)):
    kl_div_aug1=kl_div_aug1*scale_aug1
    logger.debug('kl_div_aug1: {} {} {}'.format(kl_div_aug1.min(),kl_div_aug1.max(),kl_div_aug1.mean()))

    kl_div_aug2=kl_div_aug2*scale_aug2
    logger.debug('kl_div_aug2: {} {} {}'.format(kl_div_aug2.min(),kl_div_aug2.max(),kl_div_aug2.mean()))


    loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                  torch.mean(kl_div_aug1) +
                  torch.mean(kl_div_aug2)) / 3.
    # loss_jsd=0.
    # lables=torch.hstack(targets)
    # p_clean, p_aug1, p_aug2 = F.softmax(
    #     logits_clean, dim=1), F.softmax(
    #         logits_aug1, dim=1), F.softmax(
    #             logits_aug2, dim=1)
    # p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    # loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #               F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #               F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    # p_clean, p_aug1, p_aug2 = F.softmax(
    #     logits_clean, dim=1), F.softmax(
    #         logits_aug1, dim=1), F.softmax(
    #             logits_aug2, dim=1)
    # p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()
    # loss_jsd = 8 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #               F.kl_div(p_mixture, p_aug1, reduction='batchmean')) / 2.

    # loss for features
    loss_feature=0.
    # loss_nxt = NTXentLoss()
    # # loss_nxt = pml_dist.DistributedLossWrapper(loss_nxt)
    # loss_jsd=loss_nxt(torch.vstack(logits_all), lables)



    # loss_feature=[]
    
    # features_clean=features_all[0]
    # features_aug1=features_all[1]
    # features_aug2=features_all[2]
    # for i in range(len(features_clean)):
    #   features_tmp=torch.vstack([features_clean[i].reshape(len(logits_clean),-1),
    #                              features_aug1[i].reshape(len(logits_aug1),-1),
    #                              features_aug2[i].reshape(len(logits_aug2),-1)])
    #   loss_feature.append(loss_nxt(features_tmp, lables))
    # loss_feature=torch.mean(torch.stack(loss_feature))
    
    # loss_feature=0.
    # features_clean=features_all[0]
    # features_aug1=features_all[1]
    # features_aug2=features_all[2]
    # device_keys=list(features_clean.keys())
    # feature_num=len(features_clean[device_keys[0]])
    # samples_num=features_clean[device_keys[0]][0].shape[0]
    # for i in range(feature_num):
    #   for key in device_keys:
    #     for j in range(samples_num):
    #       loss_tmp=0.
    #       # loss_tmp+=get_corr(features_clean[key][i][j,...].flatten(),features_aug1[key][i][j,...].flatten())
    #       # loss_tmp+=get_corr(features_clean[key][i][j,...].flatten(),features_aug2[key][i][j,...].flatten())
    #       loss_tmp+=F.cosine_similarity(features_clean[key][i][j,...].flatten(),features_aug1[key][i][j,...].flatten(),axis=0)
    #       loss_tmp-=F.cosine_similarity(features_clean[key][i][j,...].flatten(),features_aug2[key][i][j,...].flatten(),axis=0)
    #       loss_tmp=loss_tmp.cpu()+1#max(loss_tmp.cpu()+1,0.)
    #       loss_feature+=loss_tmp
    #     # loss_feature+=loss_tmp.cpu()+marg
    #       # loss_tmp+=torch.norm(features_clean[key][i][j,...]-features_aug1[key][i][j,...])
    #       # loss_tmp-=torch.norm(features_clean[key][i][j,...]-features_aug2[key][i][j,...])
    #     # loss_feature+=loss_tmp.cpu()
    # loss_feature=loss_feature/feature_num/samples_num/len(device_keys)/2.

    # loss_feature=loss_feature.cuda()

    return loss_pred,loss_jsd,loss_feature



if __name__ == '__main__':

  os.environ['CUDA_VISIBLE_DEVICES']='1'
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

  writer = SummaryWriter(saved_dir)


  main()
