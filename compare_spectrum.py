from asyncio.log import logger
import os
import time
import logging

from cv2 import cvtColor
import general as g
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image,ImageColor
from torch.utils.data import DataLoader
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
# from augment_and_mix import augment_and_mix
from scipy.io import savemat
from skimage.color import rgb2lab,rgb2ycbcr,rgb2luv,rgb2hsv
from sklearn.decomposition import PCA #大数据的包实现主成分分析
import joblib
from numpy.random import choice
import augmentations
augmentations.IMAGE_SIZE = 32




def get_spectrum_fft(img):
    fft2 = np.fft.fft2(img,axes=(-1,-2))
    shift2center = np.fft.fftshift(fft2,axes=(-1,-2))
    # mag=np.sqrt(shift2center.real**2+shift2center.imag**2)
    mag=np.abs(shift2center.real)
    log_shift2center = np.log(1+mag)
    return log_shift2center

def get_spectrum(imgs):
    # images_ycbcr=g.rgb_to_ycbcr(imgs)
    images_dct=g.img2dct(imgs)
    # images_dct=g.img2dct_4part(images_ycbcr)
    return images_dct

def get_spectrum_tc(imgs):
    # images_dct=g.img2dct(imgs)
    # L2s=np.linalg.norm(images_dct,axis=(-1,-2))
    imgs=torch.tensor(imgs)
    images_dct=torch.fft.fft2(imgs)
    images_dct=torch.fft.fftshift(images_dct)
    images_dct=torch.log(1+torch.abs(images_dct.real))
    return images_dct.numpy()

def cvt_color(imgs):
    # img=img.convert('RGB')
    # img=np.array(img)
    # img=rgb2ycbcr(img)    
    # img=img.transpose(2,0,1)
    imgs_ret=[]
    assert len(imgs.shape)==4
    imgs_ret=rgb2ycbcr(imgs.transpose(2,3,0,1))
    imgs_ret=imgs_ret.transpose(2,3,0,1)
    return imgs_ret/255.0

def aug(image):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  mixture_width=3
  mixture_depth=4
  aug_severity=3
  aug_list = augmentations.augmentations_std

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(torch.from_numpy(np.array(image)).float())
  depths=[]
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = np.random.randint(1,mixture_depth)
    depths.append(float(depth))
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * torch.from_numpy(np.array(image_aug))

  mixed = (1 - m) * torch.from_numpy(np.array(image)).float() + m * mix
  depth_mean=float(np.array(depths).mean())/10
  scale=(m+depth_mean)/2
  return mixed.numpy()/255.0,scale

def batch_aug(images):
    imgs_ret=[]
    for img in images:
        img=Image.fromarray(np.uint8(img.transpose(1,2,0)*255))
        img,_=aug(img)
        imgs_ret.append(np.expand_dims(img,0))
    imgs_ret=np.vstack(imgs_ret).transpose(0,3,1,2)
    return imgs_ret



# def compare_spectrum_tc(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
#         image_tmp=cvt_color(image_tmp)
#         images_clean[i,...]=image_tmp#/255.0
#     images_clean=get_spectrum_tc(images_clean)

#     # 计算损坏图像的频谱差
#     for level in range(5):
#         my_mat={}
#         for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
#             if '/'+str(level+1) not in dataset_dir:
#                 continue
#             images_crupt=np.zeros_like(images_clean)
#             crupt_name='_'.join(dataset_dir.split('/')[-3:])
#             saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#             for j,file_name in enumerate(file_names):
#                 image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#                 image_tmp=cvt_color(image_tmp)
#                 images_crupt[j,...]=image_tmp#/255.0
#             images_crupt=get_spectrum_tc(images_crupt)
#             images_diff=(images_crupt-images_clean)
#             images_diff=np.mean(images_diff,axis=0)/(images_clean.mean(axis=0)+g.epsilon)
#             g.save_images_channel(saved_dir_tmp,images_diff)

#             for k in range(images_diff.shape[0]):
#                 my_mat[crupt_name[:-2]+'_c'+str(k)]=images_diff[k,...]
#         savemat(os.path.join(saved_dir,'corruptions_'+str(level+1)+'.mat'),my_mat)

# def compare_spectrum(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
#         image_tmp=cvt_color(image_tmp)
#         images_clean[i,...]=image_tmp#/255.0
#     images_clean=get_spectrum(images_clean)

#     # 计算损坏图像的频谱差
#     my_mat={}
#     my_std={}
#     for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
#         images_crupt=np.zeros_like(images_clean)
#         crupt_name='_'.join(dataset_dir.split('/')[-3:])
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         for j,file_name in enumerate(file_names):
#             image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#             image_tmp=cvt_color(image_tmp)
#             images_crupt[j,...]=image_tmp#/255.0
#         images_crupt=get_spectrum(images_crupt)
#         images_diff=(images_crupt-images_clean)
#         images_std=images_diff.std(axis=0)/(images_clean.mean(axis=0)+g.epsilon)
#         images_diff=np.mean(images_diff,axis=0)/(images_clean.mean(axis=0)+g.epsilon)
#         g.save_images_channel(saved_dir_tmp,images_diff)

#         for k in range(images_diff.shape[0]):
#             my_mat[crupt_name+'_c'+str(k)]=images_diff[k,...]
#             my_std[crupt_name+'_c'+str(k)]=images_std[k,...]
#     savemat(os.path.join(saved_dir,'corruptions.mat'),my_mat)
#     savemat(os.path.join(saved_dir,'corruptions_std.mat'),my_std)

# def compare_spectrum_pixel(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
#         image_tmp=cvt_color(image_tmp)
#         images_clean[i,...]=image_tmp#/255.0
#     images_clean_spectrum=get_spectrum(images_clean)

#     # 计算损坏图像的频谱差
#     my_mat={}
#     my_std={}
#     for i,dataset_dir in enumerate(tqdm(dataset_setting_crupt.dataset_dir)):
#         images_crupt=np.zeros_like(images_clean)
#         crupt_name='_'.join(dataset_dir.split('/')[-3:])
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         for j,file_name in enumerate(file_names):
#             image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#             image_tmp=cvt_color(image_tmp)
#             images_crupt[j,...]=image_tmp#/255.0
#         images_diff=images_crupt-images_clean
#         images_diff=get_spectrum(images_diff)
#         images_std=images_diff.std(axis=0)/(images_clean_spectrum.mean(axis=0)+g.epsilon)
#         images_mean=images_diff.mean(axis=0)/(images_clean_spectrum.mean(axis=0)+g.epsilon)

#         for k in range(images_mean.shape[0]):
#             my_mat[crupt_name+'_c'+str(k)]=images_mean[k,...]
#             my_std[crupt_name+'_c'+str(k)]=images_std[k,...]
#     savemat(os.path.join(saved_dir,'corruptions.mat'),my_mat)
#     savemat(os.path.join(saved_dir,'corruptions_std.mat'),my_std)


# def compare_features_augmix(dataset_setting_clean,severity,width,depth,alpha,file_names,saved_dir):
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(tqdm(file_names)):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]])
#         image_tmp=image_tmp.convert('RGB')
#         image_tmp=np.array(image_tmp).transpose(2,0,1)
#         images_clean[i,...]=image_tmp/255.0
#     features_clean=get_hog(images_clean)
#     logger.info('Finish {}'.format('cleans'))

#     images_crupt=np.zeros_like(images_clean)
#     for j in tqdm(range(images_clean.shape[0])):
#         image_tmp=images_clean[j,...].transpose(1,2,0)
#         image_tmp=augment_and_mix(image_tmp,severity,width,depth,alpha)
#         images_crupt[j,...]=image_tmp.transpose(2,0,1)
#     features_crupt=get_hog(images_crupt)
#     setting_name='s'+str(severity)+'_w'+str(width)+'_d'+str(depth)+'_a'+str(alpha)
#     logger.info('Finish {}'.format(setting_name))

#     sims=get_vec_sim(features_crupt,features_clean)
#     saved_dir_tmp=os.path.join(saved_dir,'augmix')
#     if not os.path.exists(saved_dir_tmp):
#         os.makedirs(saved_dir_tmp)
#     np.savetxt(os.path.join(saved_dir_tmp,setting_name+'.txt'),sims)

# def compare_spectrum_rgb_whole(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir,type='DFT'):
#     my_mat={}
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]]).convert('RGB')
#         images_clean[i,...]=np.array(image_tmp).transpose(2,0,1)/255.0
#     if 'DFT'==type:
#         images_clean=get_spectrum_fft(images_clean)
#     elif 'DCT'==type:
#         images_clean=get_spectrum(images_clean)
#     else:
#         raise ValueError('Wrong type')
#     my_mat['spectrum']=images_clean.mean(axis=0).mean(axis=0)
#     saved_dir_tmp=os.path.join(saved_dir,type,'clean')
#     if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#     savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)


#     # 计算损坏图像的频谱差
#     for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
#         images_crupt=np.zeros_like(images_clean)
#         crupt_name='_'.join(dataset_dir.split('/')[-3:])
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         for j,file_name in enumerate(file_names):
#             image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#             images_crupt[j,...]=np.array(image_tmp).transpose(2,0,1)/255.0
#         if 'DFT'==type:
#             images_crupt=get_spectrum_fft(images_crupt)
#         elif 'DCT'==type:
#             images_crupt=get_spectrum(images_crupt)
#         else:
#             raise ValueError('Wrong type')
#         images_diff=images_crupt-images_clean

#         my_mat={}
#         my_mat['spectrum']=images_diff.mean(axis=0).mean(axis=0)
#         # my_mat['spectrum']=images_crupt.mean(axis=0).mean(axis=0)-images_clean.mean(axis=0).mean(axis=0)
#         saved_dir_tmp=os.path.join(saved_dir,type,crupt_name)
#         if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#         savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)

def compare_spectrum(images_all,saved_dir,DFT='DCT',COLOR='YCbCr',CHANNELS=3,RELATIVE='relative',PIXEL='whole'):
    if 'DFT'==DFT: flag_dft=True
    elif 'DCT'==DFT: flag_dft=False
    else: raise ValueError('Wrong input:{}'.format(DFT))

    if 'YCbCr'==COLOR: flag_ycbcr=True
    elif 'RGB'==COLOR: flag_ycbcr=False
    else: raise ValueError('Wrong input:{}'.format(COLOR))

    if 3==CHANNELS: flag_channel=True
    elif 1==CHANNELS: flag_channel=False
    else: raise ValueError('Wrong input:{}'.format(CHANNELS))
    
    if 'relative'==RELATIVE: flag_relative=True
    elif 'absolute'==RELATIVE: flag_relative=False
    else: raise ValueError('Wrong input:{}'.format(RELATIVE))
    
    if 'pixel'==PIXEL: flag_pixel=True
    elif 'whole'==PIXEL: flag_pixel=False
    else: raise ValueError('Wrong input:{}'.format(PIXEL))

    job='{}_{}_{}_{}_{}'.format(DFT,COLOR,CHANNELS,RELATIVE,PIXEL)
    logger.info(job)

    my_mat={}
    # 计算干净图像的频谱
    images_clean=images_all['clean']
    if flag_ycbcr: images_clean=cvt_color(images_clean)

    if flag_dft: images_clean_spectrum=get_spectrum_fft(images_clean)
    else: images_clean_spectrum=get_spectrum(images_clean)

    if flag_channel:
        for i,tmp in enumerate(images_clean_spectrum.mean(axis=0)): 
            my_mat['spectrum_c'+str(i)]=tmp
    else: my_mat['spectrum']=images_clean_spectrum.mean(axis=0).mean(axis=0)
    saved_dir_tmp=os.path.join(saved_dir,job,'clean')
    if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
    savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)


    # 计算损坏图像的频谱差
    for _,(name,images) in enumerate(tqdm(images_all.items())):
        if name=='clean': continue
        images_crupt=images
        crupt_name=name
        my_mat={}
        saved_dir_tmp=os.path.join(saved_dir,crupt_name)

        if flag_ycbcr: images_crupt=cvt_color(images_crupt)

        if flag_pixel: images_for_spectrum=images_crupt-images_clean
        else: images_for_spectrum=images_crupt

        if flag_dft: images_crupt_spectrum=get_spectrum_fft(images_for_spectrum)
        else: images_crupt_spectrum=get_spectrum(images_for_spectrum)

        if flag_pixel: images_diff=images_crupt_spectrum
        else: images_diff=images_crupt_spectrum-images_clean_spectrum

        if flag_relative: images_diff=images_diff/(images_clean_spectrum.mean(axis=0)+1e-8)

        images_diff=images_diff.mean(axis=0)


        if flag_channel:
            for i,tmp in enumerate(images_diff): my_mat['spectrum_c'+str(i)]=tmp
        else: my_mat['spectrum']=images_diff.mean(axis=0)
        # my_mat['spectrum']=images_crupt.mean(axis=0).mean(axis=0)-images_clean.mean(axis=0).mean(axis=0)
        saved_dir_tmp=os.path.join(saved_dir,job,crupt_name)
        if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
        savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)

# def compare_spectrum_rgb_whole_pixel(images_all,saved_dir,type='DFT',color='RGB'):
#     my_mat={}
#     # 计算干净图像的频谱
#     images_clean=images_all['clean']
#     if 'YCbCr'==color:
#         images_clean=cvtColor(images_clean)
#     if 'DFT'==type:
#         images_clean_spectrum=get_spectrum_fft(images_clean)
#     elif 'DCT'==type:
#         images_clean_spectrum=get_spectrum(images_clean)
#     else:
#         raise ValueError('Wrong type')
#     my_mat['spectrum']=images_clean_spectrum.mean(axis=0).mean(axis=0)
#     saved_dir_tmp=os.path.join(saved_dir,type,'clean')
#     if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#     savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)


#     # 计算损坏图像的频谱差
#     for name,images in images_all.items():
#         if name=='clean': continue
#         images_crupt=images
#         if 'YCbCr'==color:
#             images_crupt=cvtColor(images_crupt)
#         crupt_name=name
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         if 'DFT'==type:
#             images_crupt=get_spectrum_fft(images_crupt-images_clean)
#         elif 'DCT'==type:
#             images_crupt=get_spectrum(images_crupt-images_clean)
#         else:
#             raise ValueError('Wrong type')

#         my_mat={}
#         my_mat['spectrum']=images_crupt.mean(axis=0).mean(axis=0)
#         # my_mat['spectrum']=images_crupt.mean(axis=0).mean(axis=0)-images_clean.mean(axis=0).mean(axis=0)
#         saved_dir_tmp=os.path.join(saved_dir,type,crupt_name)
#         if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#         savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)



# def compare_spectrum_rgb_whole_pixel(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir,type='DFT'):
#     my_mat={}
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]]).convert('RGB')
#         images_clean[i,...]=np.array(image_tmp).transpose(2,0,1)/255.0
#     if 'DFT'==type:
#         images_clean_spectrum=get_spectrum_fft(images_clean)
#     elif 'DCT'==type:
#         images_clean_spectrum=get_spectrum(images_clean)
#     else:
#         raise ValueError('Wrong type')
#     my_mat['spectrum']=images_clean_spectrum.mean(axis=0).mean(axis=0)
#     saved_dir_tmp=os.path.join(saved_dir,type,'clean')
#     if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#     savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)


#     # 计算损坏图像的频谱差
#     for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
#         images_crupt=np.zeros_like(images_clean)
#         crupt_name='_'.join(dataset_dir.split('/')[-3:])
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         for j,file_name in enumerate(file_names):
#             image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#             images_crupt[j,...]=np.array(image_tmp).transpose(2,0,1)/255.0
#         if 'DFT'==type:
#             images_diff=get_spectrum_fft(images_crupt-images_clean)
#         elif 'DCT'==type:
#             images_diff=get_spectrum(images_crupt-images_clean)
#         else:
#             raise ValueError('Wrong type')
#         # images_diff=images_crupt-images_clean

#         my_mat={}
#         my_mat['spectrum']=images_diff.mean(axis=0).mean(axis=0)
#         # my_mat['spectrum']=images_crupt.mean(axis=0).mean(axis=0)-images_clean.mean(axis=0).mean(axis=0)
#         saved_dir_tmp=os.path.join(saved_dir,type,crupt_name)
#         if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#         savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)

# def compare_spectrum_YCbCr(dataset_setting_clean,dataset_setting_crupt,file_names,saved_dir):
#     my_mat={}
#     # 计算干净图像的频谱
#     image_shape=dataset_setting_clean.input_shape
#     image_num=len(file_names)
#     images_clean=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#     for i,file_name in enumerate(file_names):
#         image_tmp=Image.open(os.path.join(dataset_setting_clean.dataset_dir,'val',file_name)).resize([image_shape[1],image_shape[2]]).convert('RGB')
#         image_tmp=cvt_color(image_tmp)
#         images_clean[i,...]=image_tmp/255.0
#     images_clean=get_spectrum(images_clean)
#     images_clean=images_clean.mean(axis=0)
#     for i in range(images_clean.shape[0]):
#         my_mat['spectrum_c'+str(i)]=images_clean[i]
#     saved_dir_tmp=os.path.join(saved_dir,'clean')
#     if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#     savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)


#     # 计算损坏图像的频谱差
#     for i,dataset_dir in enumerate(dataset_setting_crupt.dataset_dir):
#         images_crupt=np.zeros([image_num,image_shape[0],image_shape[1],image_shape[2]])
#         crupt_name='_'.join(dataset_dir.split('/')[-3:])
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         for j,file_name in enumerate(file_names):
#             image_tmp=Image.open(os.path.join(dataset_dir,file_name)).resize([image_shape[1],image_shape[2]])
#             image_tmp=cvt_color(image_tmp)
#             images_crupt[j,...]=image_tmp/255.0
#         images_crupt=get_spectrum(images_crupt)
#         images_diff=images_crupt-images_clean
#         images_diff=images_diff.mean(axis=0)

#         my_mat={}
#         for i in range(images_diff.shape[0]):
#             my_mat['spectrum_c'+str(i)]=images_diff[i]
#         saved_dir_tmp=os.path.join(saved_dir,crupt_name)
#         if not os.path.exists(saved_dir_tmp): os.makedirs(saved_dir_tmp)
#         savemat(os.path.join(saved_dir_tmp,'spectrum.mat'),my_mat)



def output_names(dir_src):
    filenames=[]
    for root,dirs,files in os.walk(dir_src):
        if not files:
            continue
        sysn=root.split('/')[-1]
        for filename in files:
            if filename is None:
                print('filename is None')
            filenames.append(sysn+'/'+filename)
    pro_idx = np.random.permutation(len(filenames))
    file=open(dir_src+'.txt','w')
    for i in range(len(filenames)):
        file.write(filenames[pro_idx[i]]+'\n')
    file.close()

def get_images(datasets):
    if not isinstance(datasets,list):
        datasets=[datasets]

    imgs_all=[]
    labels_all=[]
    for dataset in datasets:
        imgs_tmp=[]
        labels_tmp=[]
        tmp_loader = torch.utils.data.DataLoader(dataset,batch_size=100,shuffle=False,drop_last=False,num_workers=16,pin_memory=False)
        for images, labels in tmp_loader:
            imgs_tmp.append(images.numpy())
            labels_tmp.append(labels.numpy())
        imgs_all.append(np.vstack(imgs_tmp))
        labels_all.append(np.vstack(labels_tmp))
    return imgs_all,labels_all


'''
设置
'''
model_name='allconv'
num_images=1000
os.environ['CUDA_VISIBLE_DEVICES']='0'
job='spectrum_aug'

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
saved_dir=os.path.join('./results',model_name+'_'+str(num_images),job)
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

'''
初始化日志系统
'''
set_level=logging.INFO
logger=logging.getLogger(name='r')
logger.setLevel(set_level)
formatter=logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

fh=logging.FileHandler(os.path.join(saved_dir,'log_mcts.log'))
fh.setLevel(set_level)
fh.setFormatter(formatter)

ch=logging.StreamHandler()
ch.setLevel(set_level)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
g.setup_seed(0)

'''
初始化模型和数据集
'''
if 'imagenet' in model_name:
    dataset_name='imagenet'
elif 'mnist' in model_name:
    dataset_name='mnist'
else:
    dataset_name='cifar-10'

# ckpt  = './models/'+model_name+'.pth.tar'
# ckpt  = './results/2022-04-04-23_20_35/checkpoint.pth.tar'
# model=g.select_model(model_name, ckpt)
data_setting_clean=g.dataset_setting(dataset_name)
data_setting_crupt=g.dataset_setting(dataset_name+'-c')

dataset_clean=g.load_dataset(dataset_name,data_setting_clean.dataset_dir,None,None,'val',num_images)
# dataset_crupt=g.load_dataset(dataset_name+'-c',data_setting_crupt.dataset_dir,None,None,'val',num_images)

images_clean,labels_clean = get_images(dataset_clean)
# images_crupt,labels_crupt = get_images(dataset_crupt)

# for i in range(len(labels_crupt)):
#     diff=labels_clean-labels_crupt[i]
#     assert(diff.max()==0)

# imgs_all={}
# imgs_all['clean']=images_clean[0]
# for i,crupt in enumerate(data_setting_crupt.corruption_types):
#     for j in range (5):
#         imgs_all[crupt.replace('/','_')+'_'+str(j+1)]=images_crupt[i*5+j]


'''
输出频谱
'''
# compare_spectrum(imgs_all,saved_dir,'DFT','RGB',1,'absolute','pixel')
# # compare_spectrum(imgs_all,saved_dir,'DFT','RGB',1,'absolute','whole')
# compare_spectrum(imgs_all,saved_dir,'DCT','RGB',1,'absolute','pixel')
# compare_spectrum(imgs_all,saved_dir,'DCT','RGB',1,'absolute','whole')
# compare_spectrum(imgs_all,saved_dir,'DCT','YCbCr',3,'absolute','whole')
# compare_spectrum(imgs_all,saved_dir,'DCT','YCbCr',3,'relative','whole')

'''
输出频谱
'''
imgs_all={}
imgs_all['clean']=images_clean[0]
images_crupt=batch_aug(images_clean[0])
imgs_all['aug']=images_crupt
compare_spectrum(imgs_all,saved_dir,'DCT','YCbCr',3,'relative','whole')