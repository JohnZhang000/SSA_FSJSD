import random
from typing import Final, cast
from scipy.fftpack import dct,idct


import torch
import numpy as np


class AddFourierNoise_dct:
    """
    Add Fourier noise to RGB channels respectively.
    This class is able to use as same as the functions in torchvision.transforms.

    Attributes:
        basis (torch.Tensor): scaled 2D Fourier basis. In the original paper, it is reperesented by 'v*U_{i,j}'.

    """

    def __init__(self, basis: torch.Tensor):
        assert len(basis.size()) == 2
        assert basis.size(0) == basis.size(1)
        self.basis: Final[torch.Tensor] = basis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        fourier_noise = self.basis.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # Sign of noise is chosen uniformly at random from {-1, 1} per channel.
        # In the original paper,this factor is prepresented by 'r'.
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return cast(torch.Tensor, torch.clamp(x + fourier_noise, min=0.0, max=1.0))

class AddFourierNoise_dct_channel:
    """
    Add Fourier noise to RGB channels respectively.
    This class is able to use as same as the functions in torchvision.transforms.

    Attributes:
        basis (torch.Tensor): scaled 2D Fourier basis. In the original paper, it is reperesented by 'v*U_{i,j}'.

    """

    def __init__(self, basis: torch.Tensor):
        assert len(basis.size()) == 3
        assert basis.size(-1) == basis.size(-2)
        self.basis: Final[torch.Tensor] = ycbcr_to_rgb(basis)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        fourier_noise = self.basis#.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # Sign of noise is chosen uniformly at random from {-1, 1} per channel.
        # In the original paper,this factor is prepresented by 'r'.
        # fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        # fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        # fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise *= random.randrange(-1, 2, 2)

        return cast(torch.Tensor, torch.clamp(x + fourier_noise, min=0.0, max=1.0))

def ycbcr_to_rgb(imgs):
    assert(len(imgs.shape)==3)
    assert(imgs.shape[-1]==imgs.shape[-2])
    
    y=imgs[0,...]
    cb=imgs[1,...]
    cr=imgs[2,...]
    
    delta=0.5
    cb_shift=cb-delta
    cr_shift=cr-delta
    
    r=y+1.403*cr_shift
    g=y-0.714*cr_shift-0.344*cb_shift
    b=y+1.773*cb_shift
    
    imgs_out=torch.zeros_like(imgs)
    imgs_out[0,...]=r
    imgs_out[1,...]=g
    imgs_out[2,...]=b
    return imgs_out

def rgb_to_ycbcr(imgs):
    assert(len(imgs.shape)==3)
    assert(imgs.shape[-1]==imgs.shape[-2])
    
    r=imgs[0,...]
    g=imgs[1,...]
    b=imgs[2,...]
    
    delta=0.5
    y=0.299*r+0.587*g+0.114*b
    cb=(b-y)*0.564+delta
    cr=(r-y)*0.713+delta
    
    imgs_out=torch.zeros_like(imgs)
    imgs_out[0,...]=y
    imgs_out[1,...]=cb
    imgs_out[2,...]=cr
    return imgs_out

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def img2dct(clean_imgs):
    assert(3==len(clean_imgs.shape))
    assert(clean_imgs.shape[-1]==clean_imgs.shape[-2])
    c = clean_imgs.shape[0]
    
    block_dct=np.zeros_like(clean_imgs)
    sign_dct=np.zeros_like(clean_imgs)
    for j in range(c):
        ch_block_cln=clean_imgs[j,:,:]                   
        ch_block_cln=dct2(ch_block_cln)
        sign_dct[j:,:]=np.sign(ch_block_cln)
        block_dct[j,:,:]=np.log(1+np.abs(ch_block_cln))
    return block_dct,sign_dct

def dct2img(dct_imgs): #此处应按exp缩放
    assert(3==len(dct_imgs.shape))
    assert(dct_imgs.shape[-1]==dct_imgs.shape[-2])
    c = dct_imgs.shape[0]

    images=np.zeros_like(dct_imgs)
    for j in range(c):
        ch_block_cln=dct_imgs[j,:,:]#-1
        # ch_block_cln=np.exp(ch_block_cln)
        # ch_block_cln[ch_block_cln<=(np.exp(-1)+1e-6)]=0
        ch_block_cln=idct2(ch_block_cln)
        images[j,:,:]=ch_block_cln
    return images
    
class AddFourierNoise_dct_channel_relative:
    """
    Add Fourier noise to RGB channels respectively.
    This class is able to use as same as the functions in torchvision.transforms.

    Attributes:
        basis (torch.Tensor): scaled 2D Fourier basis. In the original paper, it is reperesented by 'v*U_{i,j}'.

    """

    def __init__(self, spectrum: torch.Tensor, eps: float):
        assert len(spectrum.size()) == 3
        assert spectrum.size(-1) == spectrum.size(-2)
        self.spectrum: Final[np.ndarray] = spectrum.numpy()
        self.eps: Final[float] = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        # fourier_noise = self.spectrum#.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)
        
        x_ycbcr=rgb_to_ycbcr(x)
        x_dct,sign_dct=img2dct(x_ycbcr.numpy())

        base=x_dct*self.spectrum*self.eps
        fourier_noise=dct2img(base*sign_dct)
        fourier_noise=ycbcr_to_rgb(torch.from_numpy(fourier_noise))
        fourier_noise *= random.randrange(-1, 2, 2)

        return cast(torch.Tensor, torch.clamp(x + fourier_noise, min=0.0, max=1.0))
