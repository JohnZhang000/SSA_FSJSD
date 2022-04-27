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
"""Base augmentations operators."""

import enum
from statistics import variance
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
from scipy.io import loadmat
import joblib
import os

# ImageNet code should change this value
IMAGE_SIZE = 32
# IMAGE_SIZE = 224

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def AddSaltPepperNoise(pil_img, level):

    img = np.array(pil_img)                                                             # 图片转numpy
    h, w, c = img.shape
    Nd = np.random.random()#*level
    Sd = 1 - Nd
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
    mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
    img[mask == 0] = 0                                                              # 椒
    img[mask == 1] = 255                                                            # 盐
    img = Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
    return img

from scipy.fftpack import dct,idct

noise_Y=np.loadtxt('0.txt')
noise_Cb=np.loadtxt('1.txt')
noise_Cr=np.loadtxt('1.txt')

noise_Y = cv2.resize(noise_Y, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
noise_Cb = cv2.resize(noise_Cb, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
noise_Cr = cv2.resize(noise_Cr, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

noise_Y[noise_Y>0.5]=1
noise_Y[noise_Y<=0.5]=0
noise_Cb[noise_Cb>0.5]=1
noise_Cb[noise_Cb<=0.5]=0
noise_Cr[noise_Cr>0.5]=1
noise_Cr[noise_Cr<=0.5]=0

thresh=1#int(IMAGE_SIZE/3)
noise_Y[:,thresh:]=1
noise_Y[thresh:,:]=1
noise_Cb[:,thresh:]=1
noise_Cb[thresh:,:]=1
noise_Cr[:,thresh:]=1
noise_Cr[thresh:,:]=1

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def img2dct(clean_imgs):
    clean_imgs=clean_imgs.convert('YCbCr')
    clean_imgs=np.array(clean_imgs).transpose(2,0,1)
    clean_imgs=clean_imgs/255.
    
    block_dct=np.zeros_like(clean_imgs)
    sign_dct=np.zeros_like(clean_imgs)
    for j in range(clean_imgs.shape[0]):
        ch_block_cln=clean_imgs[j,:,:]                   
        ch_block_cln=dct2(ch_block_cln)
        sign_dct[j,:,:]=np.sign(ch_block_cln)
        ch_block_cln = np.log(1+np.abs(ch_block_cln))
        block_dct[j,:,:]=ch_block_cln
    block_dct=block_dct
    return block_dct,sign_dct

def dct2img(dct_imgs,sign_dct):
    block_cln=np.zeros_like(dct_imgs)
    for j in range(dct_imgs.shape[0]):
        ch_block_cln=dct_imgs[j,:,:]
        # ch_block_cln=np.power(10,ch_block_cln)-1
        ch_block_cln=np.exp(ch_block_cln)-1
        ch_block_cln=ch_block_cln*sign_dct[j,:,:]
        ch_block_cln=idct2(ch_block_cln)
        block_cln[j,:,:]=ch_block_cln
    block_cln=block_cln.transpose(1,2,0).clip(0,1)
    clean_imgs=Image.fromarray(np.uint8(block_cln*255),mode='YCbCr')
    clean_imgs=clean_imgs.convert('RGB')   
    return clean_imgs

def AddImpulseNoise(pil_img, level):
    dct,sign=img2dct(pil_img)

    dct_noise=dct.copy()
    dct_noise[0,...]=noise_Y*dct_noise[0,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*np.random.random()*level/2
    dct_noise[1,...]=noise_Y*dct_noise[1,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*np.random.random()*level/2
    dct_noise[2,...]=noise_Y*dct_noise[2,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*np.random.random()*level/2
    dct=dct+dct_noise

    img_out=dct2img(dct,sign)
    return img_out

def AddContrast(pil_img, level):
    dct,sign=img2dct(pil_img)

    dct0=dct[:,0,0]
    scale_level=np.random.random()*1.5
    dct=dct*scale_level
    dct[:,0,0]=dct0*(1-np.random.random()*0.01)
    # print(scale_level)

    img_out=dct2img(dct,sign)
    return img_out

def add_noise_on_spectrum(img_channel,radius,scale,std):
  # print('r:{} s:{}'.format(radius,scale))
  
  mask=np.ones_like(img_channel)*scale
  mask+=np.random.randn(mask.shape[0],mask.shape[1])*std

  H,W=img_channel.shape
  x, y = np.ogrid[:H, :W]
  r2= x*x+y*y
  circmask = r2 <= radius * radius
  mask[circmask] = 0

  adder=1 if np.random.uniform() < 0.5 else -1
  masked_img=img_channel*(1+adder*mask)
  return masked_img

# def my_spectrum_noiser(pil_img, level):
#   dct,sign=img2dct(pil_img)
#   dct0=dct[:,0,0]

#   low=1
#   high=int(np.sqrt(dct.shape[-1]*dct.shape[-1]+dct.shape[-2]*dct.shape[-2])/3)
#   std=level/3

#   for i,dct_tmp in enumerate(dct):
#     dct[i,...]=add_noise_on_spectrum(dct_tmp,np.random.randint(low,high),1.5*np.random.random(),std*np.random.random())
#     # dct[i,...]=dct[i,...]*np.random.randn(dct.shape[-1],dct.shape[-2])*np.random.random()
  
#   dct[:,0,0]=dct0*(1-np.random.random()/10)+dct0*np.random.random()*0.01

#   img_out=dct2img(dct,sign)
#   return img_out
  
# corruptions_mean=loadmat('corruptions.mat')
# corruptions_std=loadmat('corruptions_std.mat')
# corruptions_name=list(corruptions_mean.keys())
# corruptions_name.remove('__header__')
# corruptions_name.remove('__version__')
# corruptions_name.remove('__globals__')
dir='./results/resnet50_imagenet_100/pca'
components=[]
variances=[]
for channel in range(3):
    pca=joblib.load(os.path.join(dir,'pca_{}.pkl'.format(channel)))
    components.append(pca.components_)
    variances.append(pca.explained_variance_)


def my_spectrum_noiser(pil_img, level):
  dct,sign=img2dct(pil_img)
  c,h,w=dct.shape

  for i,dct_tmp in enumerate(dct):

    choosed_components=np.random.randint(0,len(components[i]))

    diff=components[i][choosed_components].reshape((224,224))
    diff=cv2.resize(diff, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
    dct[i,...]=dct_tmp+diff*dct_tmp*100*np.random.random()*(1+level)/3
  
  img_out=dct2img(dct,sign)
  return img_out
  
# def my_spectrum_noiser(pil_img, level):
#   dct,sign=img2dct(pil_img)
#   c,h,w=dct.shape

#   for i,dct_tmp in enumerate(dct):

#     corrupt=corruptions_name[np.random.randint(0,len(corruptions_name))]
#     mean_off=corruptions_mean[corrupt].astype(np.float32)
#     std_off=corruptions_std[corrupt].astype(np.float32)
#     mean_off= cv2.resize(mean_off, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
#     std_off= cv2.resize(std_off, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
#     mean_off=mean_off*np.random.randn()#(np.random.randn(h,w))#*level/100)*#(1+np.random.random())#
#     std_off=std_off*np.random.randn(h,w)#*level/10#(1+np.random.random())#(np.random.randn(h,w)*level/100)
#     dct[i,...]=dct_tmp+dct_tmp*mean_off*(1+std_off)#dct_tmp*std_off++mean_off*dct_tmp
  
#   img_out=dct2img(dct,sign)
#   return img_out



augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y,
    AddImpulseNoise,AddContrast#,my_spectrum_noiser
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]