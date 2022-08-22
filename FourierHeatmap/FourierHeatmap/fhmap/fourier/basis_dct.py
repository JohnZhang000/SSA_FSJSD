from typing import Iterator, cast

import torch
import torch.fft as fft
from scipy.fftpack import dct,idct


def get_spectrum_dct(
    height: int,
    width: int,
    height_ignore_edge_size: int = 0,
    width_ignore_edge_size: int = 0,
    low_center: bool = True,
) -> Iterator[torch.Tensor]:
    """Return generator of specrum matrics of 2D Fourier basis.

    Note:
        - height_ignore_edge_size and width_ignore_edge_size are used for getting subset of spectrum.
          e.g.) In the original paper, Fourier Heat Map was created for a 63x63 low frequency region for ImageNet.
        - We generate spectrum one by one to avoid waste of memory.
          e.g.) We need to generate more than 25,000 basis for ImageNet.

    Args:
        height (int): Height of spectrum.
        width (int): Width of spectrum.
        height_ignore_edge_size (int, optional): Size of the edge to ignore about height.
        width_ignore_edge_size (int, optional): Size of the edge to ignore about width.
        low_center (bool, optional): If True, returned low frequency centered spectrum.

    Yields:
        torch.Tensor: Generator of spectrum size of (H, W).

    """
    B = height * width
    indices = torch.arange(height * width)
    # if low_center:
    #     indices = torch.cat([indices[B // 2 :], indices[: B // 2]])

    # drop ignoring edges
    # indices = indices.view(height, width)
    # if height_ignore_edge_size:
    #     indices = indices[height_ignore_edge_size:-height_ignore_edge_size, :]
    # if width_ignore_edge_size:
    #     indices = indices[:, :-width_ignore_edge_size]
    # indices = indices.flatten()

    for idx in indices:
        yield torch.nn.functional.one_hot(idx, num_classes=B).view(
            height, width
        ).float()

def get_spectrum_dct_channel(
    channel: int,
    height: int,
    width: int,
    height_ignore_edge_size: int = 0,
    width_ignore_edge_size: int = 0,
    low_center: bool = True,
) -> Iterator[torch.Tensor]:
    """Return generator of specrum matrics of 2D Fourier basis.

    Note:
        - height_ignore_edge_size and width_ignore_edge_size are used for getting subset of spectrum.
          e.g.) In the original paper, Fourier Heat Map was created for a 63x63 low frequency region for ImageNet.
        - We generate spectrum one by one to avoid waste of memory.
          e.g.) We need to generate more than 25,000 basis for ImageNet.

    Args:
        height (int): Height of spectrum.
        width (int): Width of spectrum.
        height_ignore_edge_size (int, optional): Size of the edge to ignore about height.
        width_ignore_edge_size (int, optional): Size of the edge to ignore about width.
        low_center (bool, optional): If True, returned low frequency centered spectrum.

    Yields:
        torch.Tensor: Generator of spectrum size of (H, W).

    """
    B = channel * height * width
    indices = torch.arange(channel * height * width)
    # if low_center:
    #     indices = torch.cat([indices[B // 2 :], indices[: B // 2]])

    # drop ignoring edges
    # indices = indices.view(height, width)
    # if height_ignore_edge_size:
    #     indices = indices[height_ignore_edge_size:-height_ignore_edge_size, :]
    # if width_ignore_edge_size:
    #     indices = indices[:, :-width_ignore_edge_size]
    # indices = indices.flatten()

    for idx in indices:
        yield torch.nn.functional.one_hot(idx, num_classes=B).view(
            channel, height, width
        ).float()

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def dct2img(dct_imgs):
    assert(2==len(dct_imgs.shape))
    assert(dct_imgs.shape[-1]==dct_imgs.shape[-2])

    images=idct2(dct_imgs.numpy())
    return torch.from_numpy(images)

def dct2img_channel(dct_imgs):
    assert(3==len(dct_imgs.shape))
    assert(dct_imgs.shape[-1]==dct_imgs.shape[-2])

    images=[]
    for dct_img in dct_imgs:
        image=idct2(dct_img.numpy())
        images.append(torch.from_numpy(image).unsqueeze(0))
    return torch.vstack(images)

def spectrum_to_basis_dct(
    spectrum: torch.Tensor, l2_normalize: bool = True
) -> torch.Tensor:
    """Convert spectrum matrix to Fourier basis by 2D FFT. Shape of returned basis is (H, W).

    Note:
        - Currently, only supported the case H==W. If H!=W, returned basis might be wrong.
        - In order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args:
        spectrum (torch.Tensor): 2D spectrum matrix. Its shape should be (H, W//2+1).
                                 Here, (H, W) represent the size of 2D Fourier basis we want to get.
        l2_normalize (bool): If True, basis is l2 normalized.

    Returns:
        torch.Tensor: 2D Fourier basis.

    """
    assert len(spectrum.size()) == 2
    H = spectrum.size(-2)  # currently, only consider the case H==W
    # basis = fft.irfftn(spectrum, s=(H, H), dim=(-2, -1))
    basis=dct2img(spectrum)


    if l2_normalize:
        return cast(torch.Tensor, basis / basis.norm(dim=(-2, -1))[None, None])
    else:
        return cast(torch.Tensor, basis)

def spectrum_to_basis_dct_channel(
    spectrum: torch.Tensor, l2_normalize: bool = True
) -> torch.Tensor:
    """Convert spectrum matrix to Fourier basis by 2D FFT. Shape of returned basis is (H, W).

    Note:
        - Currently, only supported the case H==W. If H!=W, returned basis might be wrong.
        - In order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args:
        spectrum (torch.Tensor): 2D spectrum matrix. Its shape should be (H, W//2+1).
                                 Here, (H, W) represent the size of 2D Fourier basis we want to get.
        l2_normalize (bool): If True, basis is l2 normalized.

    Returns:
        torch.Tensor: 2D Fourier basis.

    """
    assert len(spectrum.size()) == 3
    C,H,W = spectrum.size()  # currently, only consider the case H==W
    # basis = fft.irfftn(spectrum, s=(H, H), dim=(-2, -1))
    basis=dct2img_channel(spectrum)


    if l2_normalize:
        norm=torch.clip(basis.norm(dim=(-2, -1)),1e-6)
        return cast(torch.Tensor, basis / norm[:,None, None])
    else:
        return cast(torch.Tensor, basis)



if __name__ == "__main__":
    import pathlib
    from typing import Final

    import torchvision

    height: Final = 32
    width: Final = 17
    height_ignore_edge_size: Final = 0
    width_ignore_edge_size: Final = 0
    image_size: Final = height
    padding: Final = 2

    savedir: Final = pathlib.Path("outputs")
    savedir.mkdir(exist_ok=True)

    spectrum = torch.stack(
        [
            x
            for x in get_spectrum_dct(
                height,
                width,
                height_ignore_edge_size,
                width_ignore_edge_size,
                low_center=True,
            )
        ]
    )
    basis = torch.stack(
        [spectrum_to_basis_dct(s, l2_normalize=True) * 10.0 for s in spectrum]
    )

    basis_rightside = torchvision.utils.make_grid(
        basis[:, None, :, :], nrow=width - width_ignore_edge_size, padding=padding
    )[:, (image_size + padding) :, : -(image_size + padding)]
    basis_leftside = torch.flip(basis_rightside, (-2, -1))
    all_basis = torch.cat(
        [
            basis_leftside[:, :, : -(image_size + padding)],
            basis_rightside[:, :, padding:],
        ],
        dim=-1,
    )
    torchvision.utils.save_image(
        all_basis, savedir / "basis.png", nrow=width - width_ignore_edge_size
    )
