"""Spectral index transformations for multi-spectral remote sensing imagery."""

from typing import Union
import numpy as np
import torch


def compute_ndvi(
    nir: Union[np.ndarray, torch.Tensor],
    red: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculates Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)
    Valid range is [-1.0, 1.0]. High values indicate dense green vegetation canopy.
    """
    if isinstance(nir, torch.Tensor):
        return (nir - red) / (nir + red + eps)
    return (nir - red) / (nir + red + eps)


def compute_ndwi(
    green: Union[np.ndarray, torch.Tensor],
    nir: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculates Normalized Difference Water Index (NDWI) - McFeeters formulation.

    NDWI = (Green - NIR) / (Green + NIR)
    Valid range is [-1.0, 1.0]. Positive values indicate open water bodies or high surface moisture.
    """
    if isinstance(green, torch.Tensor):
        return (green - nir) / (green + nir + eps)
    return (green - nir) / (green + nir + eps)


def compute_nbr(
    nir: Union[np.ndarray, torch.Tensor],
    swir2: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculates Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Valid range is [-1.0, 1.0]. Scars and bare exposed subsoil show marked reduction in NBR.
    """
    if isinstance(nir, torch.Tensor):
        return (nir - swir2) / (nir + swir2 + eps)
    return (nir - swir2) / (nir + swir2 + eps)


def compute_patch_indices(img: np.ndarray) -> np.ndarray:
    """Computes NDVI, NDWI, and NBR channels for a Landslide4Sense 14-band patch (H, W, 14).

    Sentinel-2 Band indices in L4S:
      B02 (Blue): 1
      B03 (Green): 2
      B04 (Red): 3
      B08 (NIR): 7
      B12 (SWIR2): 11

    Returns:
        Array of shape (H, W, 3) containing [NDVI, NDWI, NBR].
    """
    green = img[:, :, 2]
    red = img[:, :, 3]
    nir = img[:, :, 7]
    swir2 = img[:, :, 11]

    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    nbr = compute_nbr(nir, swir2)

    return np.stack([ndvi, ndwi, nbr], axis=-1)
