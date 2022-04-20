from PIL import Image
from augmentations import img2dct,dct2img,AddContrast,AddImpulseNoise
import numpy as np
import cv2
IMAGE_SIZE=224

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

noise_Y[:,75:]=1
noise_Y[75:,:]=1
noise_Cb[:,75:]=1
noise_Cb[75:,:]=1
noise_Cr[:,75:]=1
noise_Cr[75:,:]=1

img=Image.open('n01440764_18.JPEG').resize((IMAGE_SIZE,IMAGE_SIZE))

img_out=AddContrast(img,3)

# dct,sign=img2dct(img)

# level=1
# dct_noise=dct.copy()
# dct_noise[0,...]=noise_Y*dct_noise[0,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*level
# dct_noise[1,...]=noise_Y*dct_noise[1,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*level
# dct_noise[2,...]=noise_Y*dct_noise[2,...]*np.random.randn(IMAGE_SIZE,IMAGE_SIZE)*level
# dct=dct+dct_noise
# # dct0=dct[:,0:2,0:2]
# # dct=dct*1.5
# # dct[:,0:2,0:2]=dct0
img_out=AddImpulseNoise(img_out,3)
# img_out=dct2img(dct,sign)
img_out.save('3.png')

# import matplotlib.pyplot as plt

# from skimage import data
# from skimage.color import rgb2hsv

# rgb_img = data.coffee()
# hsv_img = rgb2hsv(rgb_img)
# hue_img = hsv_img[:, :, 0]
# value_img = hsv_img[:, :, 2]

# import torch

# a=torch.randn(1,3,224,224)
# a=a.cuda(2)
# a.device=torch.device('cuda')
# print(a)