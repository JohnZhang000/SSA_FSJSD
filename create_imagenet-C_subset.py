#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:14:11 2021

@author: dell
"""
import os
# import sys
from tqdm import tqdm
import shutil 
import numpy as np

def get_imgnames(folder):
    img_path_list=[]
    sysnet_list=[]
    for root,dirs,files in tqdm(os.walk(folder)):
        if not dirs:
            sysnet=root.split('/')[-1]#[1:]
            sysnet_list.append(sysnet)
        for filename in files:
            file_path = os.path.join(root, filename)
            # file_path = root+'/'+filename
            img_path_list.append(file_path)
    sysnet_list=list(set(sysnet_list))
    sysnet_list.sort()
    return img_path_list,sysnet_list


def create_subset(src_folder,dst_folder,img_path_list,sysnet_list,topk=100):
    sysnet_save_list=sysnet_list[0:topk]
    # sysnet_save_list=[str(sysnet_save[i])+'/' for i in range(len(sysnet_save))]
    
    for img_now in tqdm(img_path_list):
        sysnet_now=img_now.split('/')[-2]
        # idx_find=[sysnet_save_list[i] in img_now for i in range(len(sysnet_save_list))]
        if sysnet_now in sysnet_save_list:
            dst_name=img_now.replace(src_folder,dst_folder)
            dst_sub_folder=dst_name.replace(dst_name.split('/')[-1],'')
            if not os.path.isdir(dst_sub_folder):
                os.makedirs(dst_sub_folder)
            shutil.copy(img_now, dst_name)
            
            
            # print("%s\n"%(img_now))

    return


src_folder = '/media/ubuntu204/F/Dataset/ImageNet-C-100'
dst_folder = '/media/ubuntu204/F/Dataset/ILSVRC2012-10-C'

CORRUPTIONS = [
    'noise/gaussian_noise', 'noise/shot_noise', 'noise/impulse_noise', 
    'blur/defocus_blur', 'blur/glass_blur', 'blur/motion_blur', 'blur/zoom_blur', 
    'weather/snow', 'weather/frost', 'weather/fog', 'weather/brightness', 
    'digital/contrast', 'digital/elastic_transform', 'digital/pixelate', 'digital/jpeg_compression',
    'extra/gaussian_blur', 'extra/saturate', 'extra/spatter', 'extra/speckle_noise'
]

for corruption in CORRUPTIONS:
    for i in range(5):
        print('%s_%d'%(corruption,i))
        src_folder_tmp=os.path.join(src_folder,corruption,str(i+1))
        dst_folder_tmp=os.path.join(dst_folder,corruption,str(i+1))
        img_path_list,sysnet_list=get_imgnames(src_folder_tmp)
        create_subset(src_folder_tmp,dst_folder_tmp,img_path_list,sysnet_list,10)
