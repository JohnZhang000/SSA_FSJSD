#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:35 2021

@author: ubuntu204
"""

import os
import numpy as np
import torch
import torchvision.models as models
# from art.attacks.evasion import FastGradientMethod,DeepFool
# from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
# from art.attacks.evasion import ProjectedGradientDescent,BasicIterativeMethod
# from art.attacks.evasion import UniversalPerturbation
# from foolbox import PyTorchModel
# from foolbox.attacks import L2PGD,L2FastGradientAttack
from models.allconv import AllConvNet
from models.resnet import resnet50
from models.vgg import vgg16_bn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from scipy.fftpack import dct,idct
import socket
import PIL

attack_names=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP','FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
eps_L2=[0.1,1.0,10.0]
eps_Linf=[0.01,0.1,1.0,10.0]
epsilon=1e-10

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:35 2021

@author: ubuntu204
"""

import os
import cv2
import numpy as np
import random
import torch
import json
from anytree import Node, RenderTree,find,AsciiStyle
from anytree.exporter import JsonExporter,DotExporter

# from art.attacks.evasion import FastGradientMethod,DeepFool
# from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
# from art.attacks.evasion import ProjectedGradientDescent,BasicIterativeMethod
# from art.attacks.evasion import UniversalPerturbation
# from foolbox import PyTorchModel
# from foolbox.attacks import L2PGD,L2FastGradientAttack
from models.allconv import AllConvNet
from models.resnet import resnet50
from models.vgg import vgg16_bn
from models.lenet5 import lenet5
from torchvision import datasets
from torchvision.datasets import mnist,CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from scipy.fftpack import dct,idct
import socket
import PIL
import time
import matplotlib.pyplot as plt
import torchvision.models as models

attack_names=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP','FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
eps_L2=[0.1,1.0,10.0]
eps_Linf=[0.01,0.1,1.0,10.0]
epsilon=1e-10
      
class dataset_setting():
    def __init__(self,dataset_name='cifar-10',select_corruption_type=None,select_corruption_level=None):
        self.dataset_dir=None
        self.mean=None
        self.std=None
        self.nb_classes=None
        self.input_shape=None
        
        self.device=socket.gethostname()
        if 'estar-403'==self.device:
            self.root_dataset_dir='/home/estar/Datasets'
            self.workers=20
            self.device_num=2
        elif 'Jet'==self.device:
            self.root_dataset_dir='/mnt/sdb/zhangzhuang/Datasets'
            self.workers=32
            self.device_num=3
        elif '1080x4-1'==self.device:
            self.root_dataset_dir='/home/zhangzhuang/Datasets'
            self.workers=48
            self.device_num=2
        elif 'ubuntu204'==self.device:
            self.root_dataset_dir='/media/ubuntu204/F/Dataset'
            self.workers=48
            self.device_num=4
        else:
            raise Exception('Wrong device')

        self.workers=20
        self.device_num=2
        self.batch_size=128
        
        if 'cifar-10'==dataset_name:
            self.dataset_dir=os.path.join(self.root_dataset_dir,'Cifar-10')
            self.mean=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.std=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.nb_classes=10
            self.input_shape=(3,32,32)   
            self.batch_size=256      

    #     elif 'cifar-10-c'==dataset_name:
    #         if select_corruption_type is None:
    #             self.corruption_types=['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    #                 'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
    #                 'gaussian_noise', 'impulse_noise',  'shot_noise', 
    #                 'brightness', 'fog', 'frost','snow',
    #                 'gaussian_blur', 'saturate', 'spatter', 'speckle_noise']
    #         else:
    #             self.corruption_types=select_corruption_type

    #         if select_corruption_level is None:
    #             self.corruption_levels=range(5)
    #         else:
    #             self.corruption_levels=select_corruption_level

    #         self.dataset_dir=[]
    #         dataset_dirs_tmp=[os.path.join(self.root_dataset_dir,'ImageNet-C-100',x) for x in self.corruption_types]
    #         for dataset_dir_tmp in dataset_dirs_tmp:
    #             self.dataset_dir+=[os.path.join(dataset_dir_tmp,str(x+1)) for x in self.corruption_levels]
    #         self.mean=np.array((0.485, 0.456, 0.406),dtype=np.float32)
    #         self.std=np.array((0.229, 0.224, 0.225),dtype=np.float32)
    #         self.nb_classes=1000
    #         self.input_shape=(3,224,224)
    #         self.batch_size=256 

    #       for corruption in CORRUPTIONS:
    # # Reference to original data is mutated
    # test_data.data = np.load(base_path + corruption + '.npy')
    # test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    # test_loader_c = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=args.eval_batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True)
            
        elif 'imagenet'==dataset_name:
            self.dataset_dir=os.path.join(self.root_dataset_dir,'ILSVRC2012-100')
            self.mean=np.array((0.485, 0.456, 0.406),dtype=np.float32)
            self.std=np.array((0.229, 0.224, 0.225),dtype=np.float32)
            self.nb_classes=1000
            self.input_shape=(3,224,224)
            self.batch_size=256
        
        elif 'imagenet-c'==dataset_name:
            if select_corruption_type is None:
                self.corruption_types=['blur/defocus_blur','blur/glass_blur','blur/motion_blur','blur/zoom_blur',
                'digital/contrast','digital/elastic_transform','digital/jpeg_compression','digital/pixelate',
                'noise/gaussian_noise','noise/impulse_noise','noise/shot_noise',
                'weather/brightness','weather/fog','weather/frost','weather/snow',
                'extra/gaussian_blur','extra/saturate','extra/spatter','extra/speckle_noise']
            else:
                self.corruption_types=select_corruption_type

            if select_corruption_level is None:
                self.corruption_levels=range(5)
            else:
                self.corruption_levels=select_corruption_level

            self.dataset_dir=[]
            dataset_dirs_tmp=[os.path.join(self.root_dataset_dir,'ImageNet-C-100',x) for x in self.corruption_types]
            for dataset_dir_tmp in dataset_dirs_tmp:
                self.dataset_dir+=[os.path.join(dataset_dir_tmp,str(x+1)) for x in self.corruption_levels]
            self.mean=np.array((0.485, 0.456, 0.406),dtype=np.float32)
            self.std=np.array((0.229, 0.224, 0.225),dtype=np.float32)
            self.nb_classes=1000
            self.input_shape=(3,224,224)
            self.batch_size=256

        elif 'mnist'==dataset_name:
            self.dataset_dir=os.path.join(self.root_dataset_dir,'mnist')
            self.mean=np.array((0.0, 0.0, 0.0),dtype=np.float32)
            self.std=np.array((1.0, 1.0, 1.0),dtype=np.float32)
            self.nb_classes=10
            self.input_shape=(1,28,28)
            self.batch_size=256       

        else:
            raise Exception('Wrong dataset')
   
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed) # GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    
def select_model(model_type,dir_model):
    # dataset     ='cifar-10'
    if model_type == 'resnet50_imagenet':
        model = models.resnet50(pretrained=True).cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
    elif model_type == 'resnet50_imagenet_augmix':
        model = models.resnet50(pretrained=False).cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)
        dataset ='imagenet'
    elif model_type == 'vgg16_imagenet':
        model = models.vgg16(pretrained=True).cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
    elif model_type == 'alexnet_imagenet':
        model = models.alexnet(pretrained=True).cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
    elif model_type=='inception_v3_imagenet':
        model=models.inception_v3(pretrained=False).cuda().eval()
        dataset ='imagenet'
    elif model_type == 'resnet50':
        model = resnet50().cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)
    elif model_type == 'vgg16':
        model = vgg16_bn().cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)
    elif model_type == 'allconv':
        model = AllConvNet(10).cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)  
    elif model_type == 'lenet5_mnist':
        model = lenet5().cuda().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)  
    else:
        raise Exception('Wrong model name: {} !!!'.format(model_type))
    return model

def load_dataset(dataset,dataset_dir,dataset_mean,dataset_std,dataset_type='val',max_size=None):
    if dataset_mean is None:
        dataset_mean=np.array((0.0, 0.0, 0.0),dtype=np.float32)
    if dataset_std is None:
        dataset_std=np.array((1.0, 1.0, 1.0),dtype=np.float32)
    if 'mnist'==dataset:
        ret_datasets = datasets.mnist.MNIST(root=dataset_dir, train=('train'==dataset_type), 
                                            transform=transforms.Compose([ToTensor()]), 
                                            download=True)
    elif 'cifar-10'==dataset:
        ret_datasets = datasets.CIFAR10(root=dataset_dir, train=('train'==dataset_type), 
                                        transform=transforms.Compose([ToTensor(),
                                        transforms.Normalize(dataset_mean,dataset_std)]), 
                                        download=True)
    elif 'imagenet'==dataset:
        ret_datasets = datasets.ImageFolder(os.path.join(dataset_dir,dataset_type),
                                            transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(dataset_mean,dataset_std)
                                                ]))
    elif 'imagenet-c'==dataset:
        ret_datasets=[]
        for per_dataset_dir in dataset_dir:
            ret_dataset = datasets.ImageFolder(per_dataset_dir,
                                                transforms.Compose([
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(dataset_mean,dataset_std)
                                                    ]))
            ret_datasets.append(ret_dataset)
    else:
        raise Exception('Wrong dataset')
    
    if max_size:
        if isinstance(ret_datasets,list):
            for i,select_datasets in enumerate(ret_datasets):
                ret_datasets[i]=torch.utils.data.Subset(select_datasets, range(0,max_size))
        else:
            select_datasets=torch.utils.data.Subset(ret_datasets, range(0,max_size))
            ret_datasets=select_datasets
    return ret_datasets

def ycbcr_to_rgb(imgs):
    assert(4==len(imgs.shape))
    assert(imgs.shape[1]==imgs.shape[2])
    
    y=imgs[...,0]
    cb=imgs[...,1]
    cr=imgs[...,2]
    
    delta=0.5
    cb_shift=cb-delta
    cr_shift=cr-delta
    
    r=y+1.403*cr_shift
    g=y-0.714*cr_shift-0.344*cb_shift
    b=y+1.773*cb_shift
    
    imgs_out=np.zeros_like(imgs)
    imgs_out[...,0]=r
    imgs_out[...,1]=g
    imgs_out[...,2]=b
    return imgs_out

def rgb_to_ycbcr(imgs):
    assert(4==len(imgs.shape))
    assert(imgs.shape[2]==imgs.shape[3])
    imgs=imgs.transpose(0,2,3,1)
    
    r=imgs[...,0]
    g=imgs[...,1]
    b=imgs[...,2]
    
    delta=0.5
    y=0.299*r+0.587*g+0.114*b
    cb=(b-y)*0.564+delta
    cr=(r-y)*0.713+delta
    
    imgs_out=np.zeros_like(imgs)
    imgs_out[...,0]=y
    imgs_out[...,1]=cb
    imgs_out[...,2]=cr
    imgs_out=imgs_out.transpose(0,3,1,2)
    return imgs_out

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
    for i in range(n):
        for j in range(c):
            ch_block_cln=clean_imgs[i,:,:,j]                   
            block_cln_tmp = np.log(1+np.abs(dct2(ch_block_cln)))
            block_dct[i,:,:,j]=block_cln_tmp
    block_dct=block_dct.transpose(0,3,1,2)
    return block_dct

def img2dct_transform(clean_imgs):
    assert(4==len(clean_imgs.shape))
    assert(clean_imgs.shape[2]==clean_imgs.shape[3])
    clean_imgs=clean_imgs.transpose(0,2,3,1)
    n = clean_imgs.shape[0]
    # h = clean_imgs.shape[1]
    # w = clean_imgs.shape[2]
    c = clean_imgs.shape[3]
    
    block_dct=np.zeros_like(clean_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=clean_imgs[i,:,:,j]                   
            block_cln_tmp = dct2(ch_block_cln)
            block_dct[i,:,:,j]=block_cln_tmp
    block_dct=block_dct.transpose(0,3,1,2)
    return block_dct

def dct2img_transform(dct_imgs):
    assert(4==len(dct_imgs.shape))
    assert(dct_imgs.shape[2]==dct_imgs.shape[3])
    dct_imgs=dct_imgs.transpose(0,2,3,1)
    n = dct_imgs.shape[0]
    # h = clean_imgs.shape[1]
    # w = clean_imgs.shape[2]
    c = dct_imgs.shape[3]
    
    clean_imgs=np.zeros_like(dct_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=dct_imgs[i,:,:,j]                   
            block_cln_tmp = idct2(ch_block_cln)
            clean_imgs[i,:,:,j]=block_cln_tmp
    clean_imgs=clean_imgs.transpose(0,3,1,2)
    return clean_imgs

def img2dct_4part(clean_imgs):
    assert(4==len(clean_imgs.shape))
    assert(clean_imgs.shape[2]==clean_imgs.shape[3])
    fft2 = np.fft.fft2(clean_imgs,axes=(2,3))
    shift2center = np.fft.fftshift(fft2,axes=(2,3))
    mag=np.sqrt(shift2center.real**2+shift2center.imag**2)
    log_shift2center = np.log(1+mag)
    return log_shift2center

def save_images_channel(saved_dir,images,pre_att=None):
    assert(3==len(images.shape))
    assert(images.shape[1]==images.shape[2])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    for choosed_idx in range(images.shape[0]):
        name=str(choosed_idx) 
        saved_name=os.path.join(saved_dir,name)
        if pre_att:
            saved_name=os.path.join(saved_dir,pre_att+name)
        np.savetxt(saved_name+'.txt',images[choosed_idx,...])

        img_vanilla_tc  = images[choosed_idx,...]
        img_vanilla_np  = np.uint8(np.clip(np.round(img_vanilla_tc/img_vanilla_tc.max()*255),0,255))
        img_vanilla_np_res=cv2.resize(img_vanilla_np, (224,224))
        cv2.imwrite(saved_name+'.png', img_vanilla_np_res)