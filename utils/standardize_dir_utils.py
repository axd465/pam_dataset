# Customary Imports:
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2 as cv
import os
import datetime
import scipy
from skimage import exposure
import scipy.io as sio
import random
import shutil
import PIL
import imageio
from pathlib import Path
from PIL import Image
import utils.standardize_dir_utils
##########################################################################################################################
'''
STANDARDIZE DIR UTILS:
'''
##########################################################################################################################
# Using RGB Channels to carry fully-sampled, downsampled, and downsampling mask of grayscale data:
def pad_img_and_add_down_channel(array, downsample_ratio = [1,2], shape=[128,128], 
                                 gauss_blur_std=None, random=False, interp=False, order=3):
    '''
    This function takes in an image and outputs a three channel image, where the first channel
    is the fully-sampled image, the second channel is a downsampled version of this image 
    (with zero-fill, zero-fill + blur, or inpterpolation to retain fully sampled size), 
    and the third channel contains the downsampling binary mask
    '''
    if len(array.shape) != 0:
        if len(shape)>2:
            shape = shape[0:2]
        array = exposure.rescale_intensity(array, in_range='image', out_range=(0.0,1.0))
        down_image = np.array(array, dtype = np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        #print(full_image.shape)
        if downsample_ratio[0] == 0:
            downsample_ratio[0] = 1
        if downsample_ratio[1] == 0:
            downsample_ratio[1] = 1
        if interp:
            latent_image = array[::downsampling_ratio[0], ::downsampling_ratio[1]]
            down_image = skimage.transform.resize(latent_image, output_shape=array.shape, 
                                                  order=order, mode='reflect', cval=0, clip=True, preserve_range=True, 
                                                  anti_aliasing=True, anti_aliasing_sigma=None)
        else:
            if random:
                sparsity = 1/downsample_ratio[0] + 1/downsample_ratio[1]
                mask = np.random.random(shape) <= sparsity
            else:
                mask[::downsampling_ratio[0], ::downsampling_ratio[1]] = 1
            down_image = down_image * mask
        full_i_shape = array.shape[0]
        full_j_shape = array.shape[1]
        if full_i_shape % shape[0] != 0:
            i_left = full_i_shape%shape[0]
            i_pad = (shape[0] - i_left)//2
            rest_i = (shape[0] - i_left)%2
        else:
            i_left = 0
            i_pad = 0
            rest_i = 0
        if full_j_shape%shape[1] != 0:
            j_left = full_j_shape%shape[1]
            j_pad = (shape[1] - j_left)//2
            rest_j = (shape[1] - j_left)%2
        else:
            j_left = 0
            j_pad = 0
            rest_j = 0
        #print('i_left = '+str(i_left))
        #print('j_left = '+str(j_left))
        #print('i_pad = '+str(i_pad))
        #print('j_pad = '+str(j_pad))
        #print('rest_i = '+str(rest_i))
        #print('rest_j = '+str(rest_j))
        if gauss_blur_std is not None:
            down_image = scipy.ndimage.gaussian_filter(down_image, sigma=gauss_blur_std, order=0, 
                                                       output=None, mode='reflect', cval=0.0, truncate=6.0)
        full_image = np.zeros((full_i_shape, full_j_shape, 3), dtype = np.float32)
        full_image[...,0] = array # Target Array
        full_image[...,1] = down_image # Downsampled Array
        full_image[...,2] = mask # Mask Array - for display
        pad_image = np.pad(full_image, [(i_pad, ), (j_pad, ), (0, )], mode='constant', constant_values = 0)
        padded_multi_chan_image = np.pad(pad_image, [(0, rest_i), (0, rest_j), (0, 0)], 
                                         mode='constant', constant_values = 0)
    else:
        padded_multi_chan_image = np.array(0)
    return padded_multi_chan_image

def rgb2gray(img, use_avg = False):
    if use_avg:
        output = np.mean(img, axis = 2)
    else:
        output = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return output

def standardize_dir(input_dir='data/train/input', downsample_ratio=[1,2], 
                    standard_shape=(128, 128, 1), file_format='.tif', add_down_ratio=True, 
                    gauss_blur_std=None, random=False, interp=False, order=3):
    '''
    This function loops through an input directory and converts each file according to the
    function "pad_img_and_add_down_channel." The modified image is then saved into the 
    input directory and the original file is deleted. 
    '''
    file_list = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    for file in file_list:
        # Load Image
        filename = os.fsdecode(file)
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.npy'):
            array = np.load(filepath).astype(np.float32)
        else:
            array = imageio.imread(filepath)
            array = np.array(array, dtype=np.float32)
        if len(array.shape) == 3:
            array = rgb2gray(array)
        temp = pad_img_and_add_down_channel(array, downsample_ratio=downsample_ratio, shape=standard_shape,
                                            gauss_blur_std=gauss_blur_std, random=random, interp=interp, order=order)
        # Save Image
        if file_format == '.npy':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            if add_down_ratio:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_{downsample_ratio[0]}-{downsample_ratio[1]}')
            else:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_standard')
            new_filepath = new_filepath.with_suffix(file_format)
            os.remove(filepath)
            np.save(new_filepath, temp, allow_pickle=True, fix_imports=True)
        elif file_format == '.tif' or file_format == '.tiff':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            if add_down_ratio:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_{downsample_ratio[0]}-{downsample_ratio[1]}')
            else:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_standard')
            new_filepath = new_filepath.with_suffix(file_format)
            os.remove(filepath)
            imageio.imwrite(new_filepath, temp, file_format)
        else:
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            if add_down_ratio:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_{downsample_ratio[0]}-{downsample_ratio[1]}')
            else:
                new_filepath = Path(os.path.abspath(new_filepath) + f'_standard')
            new_filepath = new_filepath.with_suffix(file_format)
            os.remove(filepath)
            temp = exposure.rescale_intensity(temp, in_range='image', out_range=(0.0,255.0)).astype(np.uint8)
            imageio.imwrite(new_filepath, temp, file_format)
    return len(file_list)