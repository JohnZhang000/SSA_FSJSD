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

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from scipy.fftpack import dct,idct

# ImageNet code should change this value
IMAGE_SIZE = 32
IMPULSE_THRESH=1.0
CONTRAST_SCALE=1.5


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
        ch_block_cln=np.exp(ch_block_cln)-1
        ch_block_cln=ch_block_cln*sign_dct[j,:,:]
        ch_block_cln=idct2(ch_block_cln)
        block_cln[j,:,:]=ch_block_cln
    block_cln=block_cln.transpose(1,2,0).clip(0,1)
    clean_imgs=Image.fromarray(np.uint8(block_cln*255),mode='YCbCr')
    clean_imgs=clean_imgs.convert('RGB')   
    return clean_imgs

def my_spectrum_noiser(pil_img, level):
    dct,sign=img2dct(pil_img)
    C,H,W=dct.shape

    # add contrast
    dct0=dct[:,0,0]
    dct=dct*np.random.random()*CONTRAST_SCALE
    dct[:,0,0]=dct0

    # add noise
    mask=np.ones((H,W))
    x, y = np.ogrid[:H, :W]
    r2= x*x+y*y
    r_thresh=IMPULSE_THRESH
    circmask = r2 <= r_thresh*H*r_thresh*W
    mask[circmask] = 0

    dct=dct*(1+mask*np.random.randn(C,H,W)*np.random.random()*level/2)

    img_out=dct2img(dct,sign)
    return img_out

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, my_spectrum_noiser
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
