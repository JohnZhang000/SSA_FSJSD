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
"""Main script to launch AugMix training on ImageNet.

Currently only supports ResNet-50 training.

Example usage:
  `python imagenet.py <path/to/ImageNet> <path/to/ImageNet-C>`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models
from torchvision import transforms
import logging
augmentations.IMAGE_SIZE = 224
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import distributed as pml_dist
import torch.distributed as dist
import torch.utils.data.distributed
import warnings
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and
                     callable(models.__dict__[name]))
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',now)
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument(
    '--clean_data', default='/media/ubuntu204/F/Dataset/ILSVRC2012-10', metavar='DIR', help='path to clean ImageNet dataset')
parser.add_argument(
    '--corrupted_data', default='/media/ubuntu204/F/Dataset/ILSVRC2012-10-C', metavar='DIR', help='path to ImageNet-C dataset')
parser.add_argument(
    '--model',
    '-m',
    default='resnet50',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
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
    default=0.0001,
    help='Weight decay (L2 penalty).')
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
    default=1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')
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
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=16,
    help='Number of pre-fetching threads.')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--with_NTXentLoss', action='store_true',
                    help='with_NTXentLoss')

args = parser.parse_args()

CORRUPTIONS = [
    'noise/gaussian_noise', 'noise/shot_noise', 'noise/impulse_noise', 
    'blur/defocus_blur', 'blur/glass_blur', 'blur/motion_blur', 'blur/zoom_blur', 
    'weather/snow', 'weather/frost', 'weather/fog', 'weather/brightness', 
    'digital/contrast', 'digital/elastic_transform', 'digital/pixelate', 'digital/jpeg_compression',
    'extra/gaussian_blur', 'extra/saturate', 'extra/spatter', 'extra/speckle_noise'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = args.batch_size / 256.
  k = args.epochs // 3
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.learning_rate * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


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


# def compute_mce(corruption_accs):
#   """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
#   mce = 0.
#   for i in range(len(CORRUPTIONS)):
#     avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
#     ce = 100 * avg_err / ALEXNET_ERR[i]
#     mce += ce / 15
#   return mce


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

  ws = np.float32(
      np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
  m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

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


def train(net, train_loader, optimizer,args):
  """Train for one epoch."""
  net.train()
  data_ema = 0.
  batch_ema = 0.
  loss_ema = 0.
  acc1_ema = 0.
  acc5_ema = 0.

  end = time.time()
  for i, (images, targets) in enumerate(train_loader):
    # Compute data loading time
    data_time = time.time() - end
    optimizer.zero_grad()

    logits_clean,features_clean=get_preds_and_features(net,images[0].cuda())
    logits_aug1,features_aug1=get_preds_and_features(net,images[1].cuda())
    logits_aug2,features_aug2=get_preds_and_features(net,images[2].cuda())

    logits_all=[logits_clean, logits_aug1, logits_aug2]
    features_all=[features_clean,features_aug1,features_aug2]

    if args.with_NTXentLoss:
      targets_pos = targets[0].cuda()
      targets_aug = targets[1].cuda()
      targets_neg = targets[2].cuda()
      targets_all=[targets_pos,targets_aug,targets_neg]
    else:
      targets_all=targets.cuda()
      targets_pos = targets.cuda()

    loss_pred,loss_jsd,loss_feature = my_loss(logits_all,features_all,targets_all,args.with_NTXentLoss)
    loss=loss_pred+loss_jsd+loss_feature
    acc1, acc5 = accuracy(logits_clean, targets_pos, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
      #acc1=1
      #acc5=0
    loss.backward()
    lr=optimizer.param_groups[0]['lr']
    optimizer.step()

    # Compute batch computation time and update moving averages.
    batch_time = time.time() - end
    end = time.time()

    data_ema = data_ema * 0.1 + float(data_time) * 0.9
    batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
    loss_ema = loss_ema * 0.1 + float(loss) * 0.9
    acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
    acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0):
      if i % args.print_freq == 0:
        logger.info(
            'Epoch {} Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
            '{:.3f} | Train Acc5 {:.3f}'.format(epoch+1, i, len(train_loader), data_ema,
                                                batch_ema, loss_ema, acc1_ema,
                                                acc5_ema))
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
  return loss_ema, acc1_ema, batch_ema


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




def test_c(net, test_transform):
  """Evaluate network on given corrupted dataset."""
  # corruption_accs = {}
  corruption_acc1s = []
  corruption_acc5s = []
  for corruption in CORRUPTIONS:
    # print(corruption)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupted_data, corruption, str(s))
      val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, test_transform),
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=False)
      acc1, acc5 = test_my(net, val_loader)
      corruption_acc1s.append(acc1)
      corruption_acc5s.append(acc5)
      writer.add_scalar('corruption/'+corruption+'_'+str(s)+'_acc1', acc1,epoch)
      writer.add_scalar('corruption/'+corruption+'_'+str(s)+'_acc5', acc5,epoch)
      logger.info('{}_{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption,str(s), acc1, acc5))
    
  for i,corruption in enumerate(CORRUPTIONS):
    logger.info('{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption, np.mean(corruption_acc1s[5*i:5*(i+1)]), np.mean(corruption_acc5s[5*i:5*(i+1)])))
  logger.info('Corruption 15* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s[:-4*5]), np.mean(corruption_acc5s[:-4*5])))
  logger.info('Corruption 19* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s), np.mean(corruption_acc5s)))
  return np.mean(corruption_acc1s),np.mean(corruption_acc5s)


def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  args.gpu = gpu

  torch.manual_seed(1)
  np.random.seed(1)

  if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
      if args.dist_url == "env://" and args.local_rank == -1:
          args.local_rank = int(os.environ["RANK"])
      if args.multiprocessing_distributed:
          # For multiprocessing distributed training, rank needs to be the
          # global rank among all the processes
          args.local_rank = args.local_rank * ngpus_per_node + gpu
      dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                              world_size=args.world_size, rank=args.local_rank)

  if args.pretrained:
    logger.info("=> using pre-trained model '{}'".format(args.model))
    model = models.__dict__[args.model](pretrained=True)
  else:
    logger.info("=> creating model '{}'".format(args.model))
    model = models.__dict__[args.model]()

  if not torch.cuda.is_available():
        print('using CPU, this will be slow')
  elif args.distributed:
      # For multiprocessing distributed, DistributedDataParallel constructor
      # should always set the single device scope, otherwise,
      # DistributedDataParallel will use all available devices.
      if args.gpu is not None:
          torch.cuda.set_device(args.gpu)
          model.cuda(args.gpu)
          # When using a single GPU per process and per
          # DistributedDataParallel, we need to divide the batch size
          # ourselves based on the total number of GPUs we have
          args.batch_size = int(args.batch_size / ngpus_per_node)
          args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
          model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
      else:
          model.cuda()
          # DistributedDataParallel will divide and allocate batch_size to all
          # available GPUs if device_ids are not set
          model = torch.nn.parallel.DistributedDataParallel(model)
  elif args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model = model.cuda(args.gpu)
  else:
      # DataParallel will divide and allocate batch_size to all available GPUs
      # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      #     model.features = torch.nn.DataParallel(model.features)
      #     model.cuda()
      # else:
      model = torch.nn.DataParallel(model).cuda()
  
  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.decay)

  if args.resume:
    if os.path.isfile(args.resume):
          logger.info("=> loading checkpoint '{}'".format(args.resume))
          if args.gpu is None:
              checkpoint = torch.load(args.resume)
          else:
              # Map model to be loaded to specified single gpu.
              loc = 'cuda:{}'.format(args.gpu)
              checkpoint = torch.load(args.resume, map_location=loc)
          args.start_epoch = checkpoint['epoch']
          best_acc1 = checkpoint['best_acc1']
          if args.gpu is not None:
              # best_acc1 may be from a checkpoint from a different GPU
              best_acc1 = best_acc1.to(args.gpu)
          model.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Load datasets
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  train_transform = transforms.Compose(
      [transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip()])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])
  test_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      preprocess,
  ])

  traindir = os.path.join(args.clean_data, 'train')
  valdir = os.path.join(args.clean_data, 'val')
  train_dataset = datasets.ImageFolder(traindir, train_transform)
  if args.with_NTXentLoss:
    train_dataset = My_AugMixDataset(train_dataset, preprocess)
  else:
    train_dataset = AugMixDataset(train_dataset, preprocess)

  if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
      train_sampler = None
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      sampler=train_sampler)
  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, test_transform),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=False)



  # Distribute model across all visible GPUs
  # net = torch.nn.DataParallel(net).cuda()
  # cudnn.benchmark = True

  start_epoch = 0
  global epoch
  epoch=0



  if args.evaluate:
    mce1, mce5 = test_c(model, test_transform)
    # logger.info('Corruption mean * Acc@1 {:.3f} Acc@5 {:.3f}'.format(mce1, mce5))
    
    return

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write(
        'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

  best_acc = 0
  logger.info('Beginning training from epoch:{}'.format(start_epoch + 1))
  for epoch in range(start_epoch, args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)

    begin_time = time.time()
    adjust_learning_rate(optimizer, epoch)

    train_loss_ema, train_acc_ema, batch_ema = train(model, train_loader,
                                                      optimizer,args)
    test_loss, test_acc = test(model, val_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0):
      checkpoint = {
          'epoch': epoch,
          'model': args.model,
          'state_dict': model.state_dict(),
          'best_acc': best_acc,
          'optimizer': optimizer.state_dict(),
      }

      save_path = os.path.join(args.save, 'checkpoint.pth.tar')
      torch.save(checkpoint, save_path)
      if is_best:
        shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

      with open(log_path, 'a') as f:
        f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_time,
            train_loss_ema,
            100. * train_acc_ema,
            test_loss,
            100. * test_acc,
        ))

      writer.add_scalar('Total/Train Loss', train_loss_ema,epoch)
      writer.add_scalar('Total/Test Loss', train_loss_ema,epoch)
      writer.add_scalar('Total/Test Acc', 100*test_acc,epoch)

      logger.info(
          'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
          ' Test Error {4:.2f}'
          .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                  test_loss, 100 - 100. * test_acc))
  mce1, mce5 = test_c(model, test_transform)
  # logger.info('C_Acc1: {:.3f}, C_Acc5: {:.3f}'.format(mce1, mce5))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - mce1))

def get_preds_and_features(model,images):
    features_tmp = {}
    def hook(module, input, output): 
      for input_tmp in input:
        if 0==len(features_tmp.keys()) or input_tmp.device not in features_tmp.keys():
          features_tmp[input_tmp.device]=[]
        features_tmp[input_tmp.device].append(input_tmp)

    handles=[]
    handles.append(model.module.fc.register_forward_hook(hook))
    # for module in model.module.features:
    #   if 'GELU' in module._get_name():
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
    del features_tmp
    # torch.cuda.empty_cache()
    return preds,features

def my_loss(logits_all,features_all,targets,with_NTXentLoss=False):
    logits_clean=logits_all[0]
    logits_aug1=logits_all[1]
    logits_aug2=logits_all[2]
    
    # loss for pred
    if with_NTXentLoss:
      loss_pred = F.cross_entropy(logits_clean, targets[0])

      loss_jsd=0.
      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)
      p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()
      loss_jsd = 8 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean')) / 2.

      loss_feature=[]
      loss_nxt = NTXentLoss()
      # loss_nxt = pml_dist.DistributedLossWrapper(loss_nxt)
      lables=torch.hstack(targets)
      features_clean=features_all[0]
      features_aug1=features_all[1]
      features_aug2=features_all[2]
      for i in range(len(features_clean)):
        features_tmp=torch.vstack([features_clean[i].reshape(len(logits_clean),-1),
                                   features_aug1[i].reshape(len(logits_aug1),-1),
                                   features_aug2[i].reshape(len(logits_aug2),-1)])
        loss_feature.append(loss_nxt(features_tmp, lables))
      loss_feature=torch.mean(torch.stack(loss_feature))


    else:
      loss_pred = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

      loss_feature=0.
      
    return loss_pred,loss_jsd,loss_feature


if __name__ == '__main__':

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

  logger.info(args)
  # main()
  if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
  if args.with_NTXentLoss:
      logger.info('Run with with_NTXentLoss')
  else:
      logger.info('Run without with_NTXentLoss')

  if args.dist_url == "env://" and args.world_size == -1:
      args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
      # Since we have ngpus_per_node processes per node, the total world_size
      # needs to be adjusted accordingly
      args.world_size = ngpus_per_node * args.world_size
      # Use torch.multiprocessing.spawn to launch distributed processes: the
      # main_worker process function
      mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
      # Simply call main_worker function
      main_worker(args.gpu, ngpus_per_node, args)
