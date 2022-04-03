# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:33:25 2021

@author: DELL
"""
# import torch.fft
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
# import cupy as cp
import cv2
import os
from tqdm import tqdm
import sys
import torch
import random

class img_transformer:

    # 解释器初始化
    def __init__(self,fft_level,probability=0.5,start_level=20,rpl0=0,idfts_pool_size=50):
        # print('--------------------------')
        self.fft_level          = fft_level            #分解级别
        self.replace_prob       = probability          #替换的概率
        self.start_level        = start_level          #tihuanweizhi 
        self.rpl0               = rpl0                 #shifoutihuan0
        self.idfts_pool_size    = idfts_pool_size      # 池的大小
        self.idfts_pool_layers  = fft_level            #池的层数
        self.idfts_pool_idx_now = 0                    #池指针的位置
        self.masks              = self.create_mask(self.fft_level,rpl0,start_level)
        self.idfts_pool         = 0.5*torch.randn(idfts_pool_size,3,32,32) + 1j * torch.randn(idfts_pool_size,3,32,32)
        self.random_idx_len     = 10000
        self.random_idx         = torch.randint(0,self.idfts_pool_size,(self.random_idx_len,1))
        self.random_idx_p       = 0

    def create_mask(self,fft_level,rpl0, start_level): 
        assert(16%fft_level)==0, '16 should be divided by fft_level with no remainder!'    
        masks  =np.zeros((fft_level,3,32,32))
        offset = int(16/fft_level)
        r_list = [offset]*fft_level
        r_s_tl = 16
        r_s_dr = 16 -1
        for i in range(fft_level):
            img=np.zeros((32,32))
            if i >= rpl0 and i < start_level:
                r_b_tl=r_s_tl-r_list[i]
                r_b_dr=r_s_dr+r_list[i]
                if fft_level-1==i:
                    r_b_tl=0
                    r_b_dr=32
                cv2.rectangle(img,(r_b_tl,r_b_tl),(r_b_dr,r_b_dr),1,-1)
                if 0!=i:
                    cv2.rectangle(img,(r_s_tl,r_s_tl),(r_s_dr,r_s_dr),0,-1)
                r_s_tl = r_b_tl
                r_s_dr = r_b_dr
            masks[i,0,:,:]=img
            masks[i,1,:,:]=img
            masks[i,2,:,:]=img
        masks_ret=masks.sum(axis=0)
        return torch.from_numpy(masks_ret)
    
    # def img_transform_cp(self, img_in_np):
    #     # transfrom_ori
    #     img_in        = cp.asarray(img_in_np)
    #     img_fft       = cp.fft.fft2(img_in,axes=(0,1))
    #     img_iffts     = cp.fft.fftshift(img_fft,axes=(0,1))
        
    #     # select target
    #     choosed_idx   = self.random_idx[self.random_idx_p % self.random_idx_len]
    #     img_ifft_rpl  = self.idfts_pool[choosed_idx,...]
        
    #     # img_construct
    #     img_iffts_new = self.masks * img_iffts + (1-self.masks)*img_ifft_rpl
        
    #     # ifft
    #     img_iffts_i   = cp.fft.ifftshift(img_iffts_new,axes=(0,1)) 
    #     img_fft_i     = cp.fft.ifft2(img_iffts_i,axes=(0,1))
    #     img_np        = img_fft_i.real
    #     img_np        = cp.asnumpy(cp.clip(img_np,0,255)).astype('uint8')
    #     # img_pil       = Image.fromarray(img_np.astype('uint8'))
              
    #     # update pool
    #     self.idfts_pool[self.idfts_pool_idx_now,...]=img_iffts
    #     self.idfts_pool_idx_now +=1
    #     if self.idfts_pool_idx_now >=self.idfts_pool_size:
    #         self.idfts_pool_idx_now -= self.idfts_pool_size
            
    #     return img_np
    
    def img_transform_tc(self, img_in_tc):
        # transfrom_ori
        if random.random() < self.replace_prob:
            return img_in_tc
          
        # print(img_in_tc.data.max())
        img_fft       = torch.fft.fft2(img_in_tc)
        img_iffts     = torch.fft.fftshift(img_fft)
        
        # select target
        choosed_idx   = self.random_idx[self.random_idx_p % self.random_idx_len]
        img_ifft_rpl  = self.idfts_pool[choosed_idx,...].squeeze(0)
        
        # img_construct
        # print(self.masks.shape)
        # print(img_iffts.shape)
        # print(img_ifft_rpl.shape)
        img_iffts_new = self.masks * img_iffts + (1-self.masks)*img_ifft_rpl
        
        # ifft
        img_iffts_i   = torch.fft.ifftshift(img_iffts_new) 
        img_fft_i     = torch.fft.ifft2(img_iffts_i)
        img_tc        = img_fft_i.real
        img_tc        = torch.clip(img_tc,0,1)
        # img_pil       = Image.fromarray(img_np.astype('uint8'))
              
        # update pool
        self.idfts_pool[self.idfts_pool_idx_now,...]=img_iffts
        self.idfts_pool_idx_now +=1
        if self.idfts_pool_idx_now >=self.idfts_pool_size:
            self.idfts_pool_idx_now -= self.idfts_pool_size
            
        return img_tc.to(torch.float32)
    
    def batch_img_transform(self, dir_src, dir_dst, filenames):
        
        # create folder
        for i in range(len(filenames)):
            img_name = filenames[i].split('/')[-1]
            dir_dst_syn  = filenames[i].replace(dir_src, dir_dst).replace('/'+img_name, '')
            if not os.path.exists(dir_dst_syn):
                os.makedirs(dir_dst_syn)
        
        # transform img
        transformer = self.img_transform_cp
        opener = Image.open
        pro_idx = np.random.permutation(len(filenames))
        for i in tqdm(range(len(pro_idx))):#
            src_name = filenames[pro_idx[i]]
            img_pil  = opener(src_name).convert('RGB').resize((32,32))
            img_tsf  = Image.fromarray(transformer(np.array(img_pil))).convert('RGB')
            dst_name = src_name.replace(dir_src, dir_dst).replace('.JPEG','_tsf.JPEG').replace('.png','_tsf.png')
            img_tsf.save(dst_name,quality=95)
    
if __name__=='__main__':    

    if len(sys.argv)!=7:
        print('Manual Mode !!!')
        fft_level = 28
        rpl0 = 3
        probability = 1
        start_level = 10
        # dir_src='../ILSVRC2012/my_mini_dataset/train'
        dir_src = '../../Dataset/ImageNet-5/train'
        gpu_num = 2
        gpu_now = 0
        
    else:
        print('Terminal Mode !!!')
        fft_level   = int(sys.argv[1])
        rpl0        = int(sys.argv[2])
        probability = 1#float(sys.argv[3])
        start_level = int(sys.argv[3])
        dir_src     = sys.argv[4]
        gpu_num     = int(sys.argv[5])
        gpu_now     = int(sys.argv[6])
    # 验证批量处理图片
    print('Transforming %d level img with %d level0 and %d level after' %(fft_level,rpl0,start_level))
    if dir_src.endswith('/'):
        dir_src=dir_src[:-1]
    dir_dst=dir_src+'_tsf/train_f'+str(fft_level)+'_L'+str(rpl0)+'_H'+str(start_level)
    
    filenames_list=[]
    with open(dir_src+'.txt','r') as f:
        for line in f:
            filenames_list.append(os.path.join(dir_src,line.replace('\n','')))
    
    total_length=len(filenames_list)
    num_per_gpu=int(np.round(total_length/gpu_num))
    idx_start=num_per_gpu*gpu_now
    if gpu_now==gpu_num-1:
        idx_end=total_length
    else:
        idx_end=num_per_gpu*(gpu_now+1)
    filenames_now=filenames_list[idx_start:idx_end]
    
    # with cp.cuda.Device(gpu_now):
    #     my_loader = img_transformer(fft_level,probability,start_level,rpl0)
    #     my_loader.batch_img_transform(dir_src,dir_dst,filenames_now)
