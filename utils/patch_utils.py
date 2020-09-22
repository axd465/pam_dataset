# Customary Imports:
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2 as cv
import os
import multiprocessing as mp
import math
from pathlib import Path
from skimage import exposure
import imageio
import numpy as np
import tensorflow as tf
import utils.standardize_dir_utils
##########################################################################################################################
'''
PATCH UTILS:
'''
##########################################################################################################################
def save_patches(patch_size, input_dir, output_dir, file_format='.tif', delete_previous=True):
    file_list = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif delete_previous:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    for file in file_list:
        filepath = os.path.join(input_dir, file)
        if filepath.endswith('.npy'):
            array = np.load(filepath).astype(np.float32)
        else:
            array = imageio.imread(filepath).astype(np.float32)
        save_patches_sub_process(array, patch_size, file, output_dir, file_format)
def save_patches_sub_process(array, patch_size, file, output_dir, file_format):
    full_i_shape = array.shape[0]
    full_j_shape = array.shape[1]
    if full_i_shape % patch_size[0] != 0:
        i_left = full_i_shape%patch_size[0]
        i_pad = (patch_size[0] - i_left)//2
        rest_i = (patch_size[0] - i_left)%2
    else:
        i_left = 0
        i_pad = 0
        rest_i = 0
    if full_j_shape%patch_size[1] != 0:
        j_left = full_j_shape%patch_size[1]
        j_pad = (patch_size[1] - j_left)//2
        rest_j = (patch_size[1] - j_left)%2
    else:
        j_left = 0
        j_pad = 0
        rest_j = 0

    array = array[..., None]
    pad_image = np.pad(array, [(i_pad, ), (j_pad, ), (0, )], mode='constant', constant_values = 0)
    pad_image = np.pad(pad_image, [(0, rest_i), (0, rest_j), (0, 0)], mode='constant', constant_values = 0)
    image = tf.convert_to_tensor(pad_image)
    patches = tf.image.extract_patches(images=[image],
                                       sizes=[1, patch_size[0], patch_size[1], 1],
                                       strides=[1, patch_size[0], patch_size[1], 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
    patches = np.array(tf.reshape(patches, [-1, patch_size[0], patch_size[1], 1]))
    if file_format not in ['.npy', '.tif', '.tiff']:
        patches = exposure.rescale_intensity(patches, in_range='image', out_range=(0.0,255.0)).astype(np.uint8)
    for patch_id in range(patches.shape[0]):
        patch = patches[patch_id, ..., 0]
        new_filepath = os.path.join(output_dir, file)
        new_filepath = Path(new_filepath)
        new_filepath = new_filepath.with_suffix('')
        new_filepath = f'{new_filepath}_patch_{patch_id}{file_format}'
        if file_format == '.npy':
            np.save(new_filepath, patch, allow_pickle=True, fix_imports=True)
        elif file_format == '.tif' or file_format == '.tiff':
            imageio.imwrite(new_filepath, patch, file_format)
        else:
            imageio.imwrite(new_filepath, patch, file_format)