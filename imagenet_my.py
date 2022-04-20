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


def train(net, train_loader, optimizer):
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

    # targets_pos = targets[0].cuda()
    # targets_aug = targets[1].cuda()
    # targets_neg = targets[2].cuda()
    # targets_all=[targets_pos,targets_aug,targets_neg]

    targets_all=targets.cuda()
    targets_pos = targets.cuda()

    loss_pred,loss_jsd,loss_feature = my_loss(logits_all,features_all,targets_all)
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


def main():
  torch.manual_seed(1)
  np.random.seed(1)

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
  train_dataset = AugMixDataset(train_dataset, preprocess)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, test_transform),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=False)

  if args.pretrained:
    logger.info("=> using pre-trained model '{}'".format(args.model))
    net = models.__dict__[args.model](pretrained=True)
  else:
    logger.info("=> creating model '{}'".format(args.model))
    net = models.__dict__[args.model]()

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0
  global epoch
  epoch=0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      # start_epoch = checkpoint['epoch'] + 1
      # best_acc1 = checkpoint['best_acc1']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      logger.info('Model restored from epoch:', start_epoch)

  if args.evaluate:
    mce1, mce5 = test_c(net, test_transform)
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
    begin_time = time.time()
    adjust_learning_rate(optimizer, epoch)

    train_loss_ema, train_acc_ema, batch_ema = train(net, train_loader,
                                                      optimizer)
    test_loss, test_acc = test(net, val_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
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
  mce1, mce5 = test_c(net, test_transform)
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
    # del features_tmp
    # torch.cuda.empty_cache()
    return preds,features

def my_loss(logits_all,features_all,targets):
    marg=0.1
    logits_clean=logits_all[0]
    logits_aug1=logits_all[1]
    logits_aug2=logits_all[2]
    
    # loss for pred
    loss_pred = F.cross_entropy(logits_clean, targets)

    # loss_pred = F.cross_entropy(logits_clean, targets[0].cuda())
    # loss_pred += F.cross_entropy(logits_aug1, targets)
    # loss_pred += F.cross_entropy(logits_aug2, targets)
    # loss_pred = loss_pred/3.

    # loss for jsd 
    p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    # loss_jsd=0.
    # lables=torch.hstack(targets)
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
    # # # loss_nxt = pml_dist.DistributedLossWrapper(loss_nxt)
    # # loss_jsd=loss_nxt(torch.vstack(logits_all), lables)

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


  main()
