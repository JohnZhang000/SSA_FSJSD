from PIL import Image
import augmentations
from augmentations import img2dct,dct2img,AddContrast,AddImpulseNoise,my_spectrum_noiser
import numpy as np
import cv2
import augmentations as aug
from augmentations import dct2img,img2dct
from scipy.io import loadmat
import os
import joblib
import matplotlib.pyplot as plt

IMAGE_SIZE=224

# noise_Y=np.loadtxt('0.txt')
# noise_Cb=np.loadtxt('1.txt')
# noise_Cr=np.loadtxt('1.txt')

# noise_Y = cv2.resize(noise_Y, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
# noise_Cb = cv2.resize(noise_Cb, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
# noise_Cr = cv2.resize(noise_Cr, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
# noise_Y[noise_Y>0.5]=1
# noise_Y[noise_Y<=0.5]=0
# noise_Cb[noise_Cb>0.5]=1
# noise_Cb[noise_Cb<=0.5]=0
# noise_Cr[noise_Cr>0.5]=1
# noise_Cr[noise_Cr<=0.5]=0

# noise_Y[:,75:]=1
# noise_Y[75:,:]=1
# noise_Cb[:,75:]=1
# noise_Cb[75:,:]=1
# noise_Cr[:,75:]=1
# noise_Cr[75:,:]=1
# def aug(image):
#   """Perform AugMix augmentations and compute mixture.

#   Args:
#     image: PIL.Image input image
#     preprocess: Preprocessing function which should return a torch tensor.

#   Returns:
#     mixed: Augmented and mixed image.
#   """
#   aug_list = augmentations.augmentations
  
#   ws = np.float32(np.random.dirichlet([1] * 3))
#   m = np.float32(np.random.beta(1, 1))

#   mix = np.zeros_like(np.array(image),dtype=np.float32)
#   for i in range(3):
#     image_aug = image.copy()
#     depth = np.random.randint(
#         1, 10)
#     for _ in range(depth):
#       op = np.random.choice(aug_list)
#       image_aug = op(image_aug, 3)
#     # Preprocessing commutes since all coefficients are convex
#     mix += ws[i] * np.array(image_aug)/255.0
 
#   mixed = mix#np.array(my_spectrum_noiser(image,np.random.randint(1, 10)))/255.0
# #   mixed = (1 - m) * np.array(image)/255.0 + m * mix
#   mixed=Image.fromarray(np.uint8(mixed*255))
#   return mixed

# def add_noise_on_spectrum(imgs_spectrum):
#   # print('r:{} s:{}'.format(radius,scale))
#   c,h,w=imgs_spectrum.shape
#   for i,img_channel in enumerate(imgs_spectrum):
#     low=1
#     high=int(h/3)
#     radius=np.random.uniform(low,high)
#     scale=np.random.uniform()
#     std=np.random.uniform(0,1)


#     mask=np.ones_like(img_channel)*scale
#     mask+=np.random.randn(h,w)*scale*std

#     # H,W=img_channel.shape
#     x, y = np.ogrid[:h, :w]
#     r2= x*x+y*y
#     circmask = r2 <= radius * radius
#     mask[circmask] = 0

#     adder=1 if np.random.uniform() < 0.5 else -1
#     imgs_spectrum[i,...]=img_channel*(1+adder*mask)
#   return imgs_spectrum

# def spectrum_mix(pil_img):
#   dct,sign=img2dct(pil_img)

#   ws = np.float32(np.random.dirichlet([1] * 3))
#   m = np.float32(np.random.beta(1, 1))

#   mix = np.zeros_like(dct)
#   for i in range(3):
#     dct_aug = dct.copy()
#     depth = 3# if args.mixture_depth > 0 else np.random.randint(1, 10)
#     for _ in range(depth):
#       image_aug = add_noise_on_spectrum(dct_aug)
#     # Preprocessing commutes since all coefficients are convex
#     mix += ws[i] * image_aug
#   # mix = (1 - m) * dct + m * mix
#   img_out=dct2img(mix,sign)
#   # img_out=preprocess(img_out)
#   return img_out



img=Image.open('n01440764_18.JPEG').resize((IMAGE_SIZE,IMAGE_SIZE))
aug.THRESH=0.5
# img_out=my_spectrum_noiser(img,3)
# img_out=spectrum_mix(img)
level=np.random.randint(0,10)
print(level)
img_out=my_spectrum_noiser(img,level)

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
# img_out=AddImpulseNoise(img_out,3)
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

# dir='./results/resnet50_imagenet_100/pca'
# cleans=np.load(os.path.join(dir,'spectrum_clean.npy'))
# cleans_mean=cleans.mean(axis=0)

# for channel in range(3):
#     pca=joblib.load(os.path.join(dir,'pca_{}.pkl'.format(channel)))
#     V = pca.components_
#     V.shape

#     fig, axes = plt.subplots(3,2,figsize=(224,224),subplot_kw = {"xticks":[],"yticks":[]})
#     for i, ax in enumerate(axes.flat):
#         if i>=len(V): break
#         v_tmp=V[i,:].reshape(224,224)
#         ax.imshow(v_tmp,cmap="gray")
#         np.savetxt(os.path.join(dir,'Img_pca_{}_{}.txt'.format(channel,i)),v_tmp)
#     plt.show()
#     plt.savefig(os.path.join(dir,'Img_pca_{}_{}.png'.format(channel,i)))
#     plt.close()

#     fig, axes = plt.subplots(3,2,figsize=(224,224),subplot_kw = {"xticks":[],"yticks":[]})
#     for i, ax in enumerate(axes.flat):
#         if i>=len(V): break
#         v_tmp=V[i,:].reshape(224,224)*cleans_mean[channel,...]
#         ax.imshow(v_tmp,cmap="gray")
#         np.savetxt(os.path.join(dir,'Img_mean_pca_{}_{}.txt'.format(channel,i)),v_tmp)
#     plt.show()
#     plt.savefig(os.path.join(dir,'Img_pca_{}_{}.png'.format(channel,i)))
#     plt.close()

    




