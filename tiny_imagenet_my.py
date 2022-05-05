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
import general as g
import socket
from torch.utils.tensorboard import SummaryWriter
augmentations.IMAGE_SIZE = 64

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',now)
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and
                     callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument(
    '--clean_data', default='', metavar='DIR', help='path to clean ImageNet dataset')
parser.add_argument(
    '--corrupted_data', default='', metavar='DIR_C', help='path to ImageNet-C dataset')
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
    '--batch-size', '-b', type=int, default=256, help='Batch size.')
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
    default=50,
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
    default=5.0,
    help='r thresh for impulse noise')

args = parser.parse_args()
augmentations.IMPULSE_THRESH = args.imp_thresh
augmentations.CONTRAST_SCALE = args.contrast_scale
TOPK=args.topk
TOPk_EPOCH=int(args.topk_epoch*args.epochs)
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
  """Computes the accuracy over the k top predictions for the specified values of k."""
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


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce


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
        1, 10)
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

def select_topk(p_y,y):
    target_onehot=torch.zeros_like(p_y).scatter_(1, y.reshape(-1,1), 1)
    target_logits=torch.sum(p_y*target_onehot,axis=1)
    n_correct=torch.sum((target_logits>0.5))
    topk_aug1=target_logits.topk(int(TOPK*(len(y)-n_correct)+n_correct),dim=0)[0][-1]
    topk=target_logits>topk_aug1
    return topk

def train(net, train_loader, optimizer,scheduler):
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

    logits_clean,features_clean=get_preds_and_features(net,images[0].cuda(),args.model)
    logits_aug1,features_aug1=get_preds_and_features(net,images[1].cuda(),args.model)
    logits_aug2,features_aug2=get_preds_and_features(net,images[2].cuda(),args.model)

    targets=targets.cuda()
    # loss for pred
    n_img=len(targets)
    loss_pred = F.cross_entropy(logits_clean, targets)

    # loss for jsd
    loss_jsd=0.
    p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    # with feature similarity
    if not args.no_fsim:
      features_clean=features_clean.reshape(n_img,-1)
      features_aug1=features_aug1.reshape(n_img,-1)
      features_aug2=features_aug2.reshape(n_img,-1)

      features_clean_mean=features_clean#torch.mean(features_clean,dim=0).reshape(1,-1).repeat_interleave(n_img,dim=0)
      sim_aug1=F.cosine_similarity(features_clean_mean,features_aug1,axis=-1).cuda()
      sim_aug2=F.cosine_similarity(features_clean_mean,features_aug2,axis=-1).cuda()
      scale_aug1=torch.exp(args.alpha*(1-sim_aug1))
      scale_aug2=torch.exp(args.alpha*(1-sim_aug2))

      # if not args.no_topk:
      #   # with topk
      #   topk_aug1=select_topk(p_aug1,targets)
      #   topk_aug2=select_topk(p_aug2,targets)
      #   if not args.no_timei:
      #     # with time invariant
      #     scale_epoch=np.exp(min(0,TOPk_EPOCH-epoch))
      #   else:
      #     scale_epoch=1
      #   scale_aug1=scale_aug1*((~topk_aug1)*scale_epoch+topk_aug1)
      #   scale_aug2=scale_aug2*((~topk_aug2)*scale_epoch+topk_aug2)

      kl_div_aug1=scale_aug1*F.kl_div(p_mixture, p_aug1, reduction='none').mean(axis=-1)
      kl_div_aug2=scale_aug2*F.kl_div(p_mixture, p_aug2, reduction='none').mean(axis=-1)
    else:
      kl_div_aug1=F.kl_div(p_mixture, p_aug1, reduction='none').mean(axis=-1)
      kl_div_aug2=F.kl_div(p_mixture, p_aug2, reduction='none').mean(axis=-1)

    loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                  torch.mean(kl_div_aug1) +
                  torch.mean(kl_div_aug2)) / 3.

    loss=loss_pred+loss_jsd

    acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking

    loss.backward()
    lr=optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()

    # Compute batch computation time and update moving averages.
    batch_time = time.time() - end
    end = time.time()

    data_ema = data_ema * 0.1 + float(data_time) * 0.9
    batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
    loss_ema = loss_ema * 0.1 + float(loss) * 0.9
    acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
    acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

    if i % args.print_freq == 0:
      writer.add_scalar('Lr',lr,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Train Loss',loss_ema,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_sum',loss,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_pred',loss_pred,epoch*len(train_loader)+i)
      writer.add_scalar('Loss/Loss_jsd',loss_jsd,epoch*len(train_loader)+i)

      # for tag, value in net.named_parameters():
      #   tag = tag.replace('.', '/')
      #   writer.add_histogram(tag, value.data.cpu().numpy(), epoch*len(train_loader)+i)
      #   writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch*len(train_loader)+i)

      logger.info(
          'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
          '{:.3f} | Train Acc5 {:.3f}'.format(i, len(train_loader), data_ema,
                                              batch_ema, loss_ema, acc1_ema,
                                              acc5_ema))

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
  corruption_accs = {}
  corruption_acc1s = []
  corruption_acc5s = []
  for corruption in CORRUPTIONS:
    # print(c)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupted_data, corruption, str(s))
      val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, test_transform),
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=True)
      acc1, acc5 = test_my(net, val_loader)
      corruption_acc1s.append(acc1)
      corruption_acc5s.append(acc5)
      if corruption in corruption_accs:
        corruption_accs[corruption].append(acc1)
      else:
        corruption_accs[corruption] = [acc1]

      writer.add_scalar('corruption/'+corruption+'_'+str(s)+'_acc1', acc1,epoch)
      writer.add_scalar('corruption/'+corruption+'_'+str(s)+'_acc5', acc5,epoch)
      logger.info('{}_{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption,str(s), acc1, acc5))
  for i,corruption in enumerate(CORRUPTIONS):
    logger.info('{} * Acc@1 {:.3f} Acc@5 {:.3f}'.format(corruption, np.mean(corruption_acc1s[5*i:5*(i+1)]), np.mean(corruption_acc5s[5*i:5*(i+1)])))
  logger.info('Corruption 15* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s[:-4*5]), np.mean(corruption_acc5s[:-4*5])))
  logger.info('Corruption 19* Acc@1 {:.3f} Acc@5 {:.3f}'.format(np.mean(corruption_acc1s), np.mean(corruption_acc5s)))
  return corruption_accs


def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  train_transform = transforms.Compose(
      [transforms.RandomResizedCrop(64),
       transforms.RandomHorizontalFlip()])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])
  test_transform = transforms.Compose([
      transforms.Resize(74),
      transforms.CenterCrop(64),
      preprocess,
  ])

  device=socket.gethostname()
  if 'estar-403'==device: root_dataset_dir='/home/estar/Datasets/tiny-imagenet'
  elif 'Jet'==device: root_dataset_dir='/mnt/sdb/zhangzhuang/Datasets/tiny-imagenet'
  elif '1080x4-1'==device: root_dataset_dir='/home/zhangzhuang/Datasets/tiny-imagenet'
  elif 'ubuntu204'==device: root_dataset_dir='/media/ubuntu204/F/Dataset/tiny-imagenet'
  else: raise Exception('Wrong device')
  args.clean_data=root_dataset_dir
  args.corrupted_data=root_dataset_dir+'-c'

  traindir = os.path.join(args.clean_data, 'train')
  valdir = os.path.join(args.clean_data, 'val')
  train_dataset = datasets.ImageFolder(traindir, train_transform)
  train_dataset = AugMixDataset(train_dataset, preprocess)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers)
  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, test_transform),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers)

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
  scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                 total_steps=args.epochs*len(train_loader),
                                                 max_lr=args.learning_rate*args.batch_size / 256,
                                                 three_phase=False)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0
  global epoch
  epoch=0
  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc1 = checkpoint['best_acc1']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      logger.info('Model restored from epoch:', start_epoch)

  if args.evaluate:
    test_loss, test_acc1 = test(net, val_loader)
    logger.info('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
        test_loss, 100 * test_acc1))

    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
      logger.info('\t'.join([c] + map(str, corruption_accs[c])))

    logger.info('mCE (normalized by AlexNet): ', compute_mce(corruption_accs))
    return

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          'imagenet_{}_training_log.csv'.format(args.model))
  with open(log_path, 'w') as f:
    f.write(
        'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

  best_acc1 = 0
  logger.info('Beginning training from epoch:{}'.format(start_epoch + 1))
  for epoch in range(start_epoch, args.epochs):
    # adjust_learning_rate(optimizer, epoch)

    train_loss_ema, train_acc1_ema, batch_ema = train(net, train_loader,
                                                      optimizer,scheduler)
    test_loss, test_acc1 = test(net, val_loader)

    is_best = test_acc1 > best_acc1
    best_acc1 = max(test_acc1, best_acc1)
    checkpoint = {
        'epoch': epoch,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          batch_ema,
          train_loss_ema,
          100. * train_acc1_ema,
          test_loss,
          100. * test_acc1,
      ))

    writer.add_scalar('Total/Train Loss', train_loss_ema,epoch)
    writer.add_scalar('Total/Test Loss', train_loss_ema,epoch)
    writer.add_scalar('Total/Test Acc', 100*test_acc1,epoch)

    logger.info(
        'Epoch {:3d} | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc1 '
        '{:.2f}'
        .format((epoch + 1), train_loss_ema, test_loss, 100. * test_acc1))

  corruption_accs = test_c(net, test_transform)
  for c in CORRUPTIONS:
    logger.info('\t'.join(map(str, [c] + corruption_accs[c])))

  # logger.info('mCE (normalized by AlexNet):', compute_mce(corruption_accs))

def get_preds_and_features(model,images,model_name):
    features_tmp = {}
    def hook(module, input, output): 
      if output.device not in features_tmp.keys():
        features_tmp[output.device]=[]
      features_tmp[output.device].append(output)

    handles=[]
    if 'resnet50'== model_name: handles.append(model.module.layer4.register_forward_hook(hook))
    else: raise Exception('{} not supported'.format(model_name))


    preds = model(images)
    for handle in handles:
      handle.remove()

    # gather feature to the same device
    features=[]
    device_keys=list(features_tmp.keys())
    for key in device_keys:
        features.append(features_tmp[key][0].cuda(0))
    features=torch.cat(features,dim=0)
    return preds,features



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

  logger.info(args)

  writer = SummaryWriter(saved_dir)


  main()
