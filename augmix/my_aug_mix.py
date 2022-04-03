#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:58:04 2021

@author: dell
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from augment_and_mix import augment_and_mix


def batch_img_aug(dir_src, dir_dst):
    # search all files
    img_path_list = []
    folder_list = []
    for root,dirs,files in os.walk(dir_src):
        folder_list.append(root)
        for filename in files:
#                file_path = os.path.join(root, filename)
            file_path = root+'//'+filename
            img_path_list.append(file_path)
    pro_idx = np.random.permutation(len(img_path_list))
    # print(len(folder_list))
    
    # create folder
    for i in range(len(folder_list)):
        dst_dir  = folder_list[i].replace(dir_src, dir_dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        else:
            return 0
    
    # transform img
    opener = Image.open
    for i in tqdm(range(len(pro_idx))):#
        src_name = img_path_list[pro_idx[i]]
        img_pil  = opener(src_name).resize((224,224)).convert('RGB')
        img_np   = np.array(img_pil,dtype=np.float32)
        img_aug  = augment_and_mix(img_np)
        img_save = Image.fromarray(np.uint8(np.clip(img_aug,0,255)))
        dst_name = src_name.replace(dir_src, dir_dst)
        img_save.save(dst_name)
        
dir_src='./vanilla'        
dir_dst='./vanilla_aug'        
batch_img_aug(dir_src,dir_dst)

