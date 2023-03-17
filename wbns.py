"""
Wavelet-based Background Subtraction in 3D Fluorescence Microscopy

Author: Manuel Hüpfel, Institute of Apllied Physics, KIT, Karlsruhe, Germany
Mail: manuel.huepfel@kit.edu

Source:
Hüpfel M, Yu Kobitski A, Zhang W, Nienhaus GU.
Wavelet-based background and noise subtraction
for fluorescence microscopy images. Biomed Opt
Express. 2021 Jan 22;12(2):969-980. 
doi: 10.1364/BOE.413181. PMID: 33680553;
PMCID: PMC7901331.

Github:
https://github.com/NienhausLabKIT/HuepfelM/tree/master/WBNS
"""

# import required packages
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
import tifffile as tif
from pywt import wavedecn, waverecn
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from skimage import io
import os

# insert resolution in units of pixels (FWHM of the PSF)
resolution_px = 4
# insert the noise level. If resolution_px > 6 then noise_lvl = 2 may be better
noise_lvl = 1  # default = 1


def wavelet_based_BG_subtraction(image, num_levels, noise_lvl):

    coeffs = wavedecn(image, 'db1', level=None)  # decomposition
    coeffs2 = coeffs.copy()

    for BGlvl in range(1, num_levels):
        # set lvl 1 details  to zero
        coeffs[-BGlvl] = {k: np.zeros_like(v)
                          for k, v in coeffs[-BGlvl].items()}

    Background = waverecn(coeffs, 'db1')  # reconstruction
    del coeffs
    BG_unfiltered = Background
    # gaussian filter sigma = 2^#lvls
    Background = gaussian_filter(Background, sigma=2**num_levels)

    coeffs2[0] = np.ones_like(coeffs2[0])  # set approx to one (constant)
    for lvl in range(1, np.size(coeffs2)-noise_lvl):
        # keep first detail lvl only
        coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()}
    Noise = waverecn(coeffs2, 'db1')  # reconstruction
    del coeffs2

    return Background, Noise, BG_unfiltered


def wbns(file_path, filename, save_dir):
    # number of levels for background estimate
    num_levels = np.uint16(np.ceil(np.log2(resolution_px)))

    # read image file adjust shape if neccessary (padding) and plot
    image = io.imread(os.path.join(file_path, filename))
    img_type = image.dtype
    print(img_type.name)
    image = np.array(image, dtype='float32')

    #image = np.array(io.imread(os.path.join(data_dir, file)),dtype = 'float32')
    if np.ndim(image) == 2:
        shape = np.shape(image)
        image = np.reshape(image, [1, shape[0], shape[1]])
    shape = np.shape(image)
    if shape[1] % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 1), (0, 0)), 'edge')
        pad_1 = True
    else:
        pad_1 = False
    if shape[2] % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 0), (0, 1)), 'edge')
        pad_2 = True
    else:
        pad_2 = False

    # extract background and noise
    num_cores = multiprocessing.cpu_count()  # number of cores on your CPU
    res = Parallel(n_jobs=num_cores, max_nbytes=None)(delayed(wavelet_based_BG_subtraction)(
        image[slice], num_levels, noise_lvl) for slice in range(np.size(image, 0)))
    Background, Noise, BG_unfiltered = zip(*res)

    # convert to float64 numpy array
    Noise = np.asarray(Noise, dtype='float32')
    Background = np.asarray(Background, dtype='float32')
    BG_unfiltered = np.asarray(BG_unfiltered, dtype='float32')

    # undo padding
    if pad_1:
        image = image[:, :-1, :]
        Noise = Noise[:, :-1, :]
        Background = Background[:, :-1, :]
        BG_unfiltered = BG_unfiltered[:, :-1, :]
    if pad_2:
        image = image[:, :, :-1]
        Noise = Noise[:, :, :-1]
        Background = Background[:, :, :-1]
        BG_unfiltered = BG_unfiltered[:, :, :-1]

    # subtract BG only
    result = image - Background
    result[result < 0] = 0  # positivity constraint

    # save result
    result = np.asarray(result, dtype=img_type.name)
    tif.imsave(os.path.join(save_dir, filename), result, bigtiff=False)


wbns("datasets/Laloli_et_all2022_raw_images/not_infected/H1_33_ICV_MC_1_GFP.tif",
     "", "test_wbns.tif")
