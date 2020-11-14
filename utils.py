
from __future__ import division

import copy
import math
import os,random
import numpy as np
import scipy.misc
from tensorflow.contrib import slim
import tensorflow as tf
from PIL import Image

from glob import glob


############################################################
# data loader
############################################################

def load_data_preup(pair_paths):
    left_list=[]
    right_list=[]
    left_lr_list=[]
    right_lr_list=[]

    for i in range(len(pair_paths)):

        left = Image.open(pair_paths[i][0])
        w,h = left.size

        left_lr_2 = left.resize((w//2, h//2), Image.BICUBIC)
        left_lr_2 = left_lr_2.resize((w, h), Image.BICUBIC)

        left_lr_3 = left.resize((w//3, h//3), Image.BICUBIC)
        left_lr_3 = left_lr_3.resize((w, h), Image.BICUBIC)

        left_lr_4 = left.resize((w//4, h//4), Image.BICUBIC)
        left_lr_4 = left_lr_4.resize((w, h), Image.BICUBIC)
        
        right = Image.open(pair_paths[i][1])

        right_lr_2 = right.resize((w//2, h//2), Image.BICUBIC)
        right_lr_2 = right_lr_2.resize((w, h), Image.BICUBIC)

        right_lr_3 = right.resize((w//3, h//3), Image.BICUBIC)
        right_lr_3 = right_lr_3.resize((w, h), Image.BICUBIC)

        right_lr_4 = right.resize((w//4, h//4), Image.BICUBIC)
        right_lr_4 = right_lr_4.resize((w, h), Image.BICUBIC)

        if np.random.rand()>=0.5:
            left = left.transpose(Image.FLIP_LEFT_RIGHT) 
            left_lr_2 = left_lr_2.transpose(Image.FLIP_LEFT_RIGHT) 
            left_lr_3 = left_lr_3.transpose(Image.FLIP_LEFT_RIGHT) 
            left_lr_4 = left_lr_4.transpose(Image.FLIP_LEFT_RIGHT) 

            right = right.transpose(Image.FLIP_LEFT_RIGHT) 
            right_lr_2 = right_lr_2.transpose(Image.FLIP_LEFT_RIGHT) 
            right_lr_3 = right_lr_3.transpose(Image.FLIP_LEFT_RIGHT) 
            right_lr_4 = right_lr_4.transpose(Image.FLIP_LEFT_RIGHT) 


        left = np.array(left).astype(np.float32)
        left_lr_2 = np.array(left_lr_2).astype(np.float32)
        left_lr_3 = np.array(left_lr_3).astype(np.float32)
        left_lr_4 = np.array(left_lr_4).astype(np.float32)

        right = np.array(right).astype(np.float32)
        right_lr_2 = np.array(right_lr_2).astype(np.float32) 
        right_lr_3 = np.array(right_lr_3).astype(np.float32) 
        right_lr_4 = np.array(right_lr_4).astype(np.float32) 

        left_list.append(left)
        left_list.append(left)
        left_list.append(left)

        right_list.append(right)
        right_list.append(right)
        right_list.append(right)

        left_lr_list.append(left_lr_2)
        left_lr_list.append(left_lr_3)
        left_lr_list.append(left_lr_4)

        right_lr_list.append(right_lr_2)
        right_lr_list.append(right_lr_3)
        right_lr_list.append(right_lr_4)

    return np.array(left_list)/255., np.array(right_list)/255., np.array(left_lr_list)/255., np.array(right_lr_list)/255.

def load_stereo_images_preup(paths, scale_factor=2):
    out_list = []
    for path in paths:
        hr = imread(path)  # high resolution depth
        if scale_factor%2==0:
            hr = modcrop(hr, scale_factor)
        else:
            hr = modcrop(hr, scale_factor*2)
        h,w,c = hr.shape
        
        lr = scipy.misc.imresize(hr, [int(h//scale_factor), int(w//scale_factor)], interp='bicubic')
        lr = scipy.misc.imresize(lr, [h, w], interp='bicubic') / 255.
        out_list += [hr, lr]
    return out_list
############################################################
# image ops
############################################################

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge(images, size):
    """merge all the images within a batch and arrage them in size[0] rows and size[1] columns

    Parameters
    ----------
    images: batch images
    size: [rows,columns]

    Returns
    -------
    return the integrated image
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]  #
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def modcrop(img, scale_factor):
    if img.ndim == 2:
        h, w = img.shape
        return img[0:int((h//scale_factor)*scale_factor), 0:int((w//scale_factor)*scale_factor)]
    elif img.ndim == 3:
        h, w, c = img.shape
        return img[0:int((h//scale_factor)*scale_factor), 0:int((w//scale_factor)*scale_factor),:]
    else:
        raise TypeError("dimension of img must be 2 or 3")


############################################################
# tools
############################################################
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    """Check if the target folder exists, if not, create one
    
    Arguments:
        log_dir {[str]} -- name of target folder
    
    Returns:
        [str] -- return the name of target folder
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    """Convert str to boolean, used in argparse
    """
    return x.lower() in ('true')

def cal_conv_pad(h,w,s,k):
    """ calculate padding of 'SAME'
    
    Arguments:
        h {[int]} -- height
        w {[int]} -- widht
        s {[int]} -- stride
        k {[int]} -- kernel size
    Returns:
        [int] -- pad_top, pad_down, pad_left, pad_right
    """
    n_h = np.ceil(h/s)
    n_w = np.ceil(w/s)
    p_h = (n_h-1) * s + k -h
    p_w = (n_w-1) * s + k -w

    p_top = p_h//2
    p_down = p_h - p_top

    p_left = p_w//2
    p_right = p_w - p_left

    return int(p_top), int(p_down), int(p_left), int(p_right)

def cal_deconv_pad(h,w,s,k,out_h,out_w):
    """calculate deconv pad
    
    Arguments:
        h {[int]} -- height
        w {[int]} -- widht
        s {[int]} -- stride
        k {[int]} -- kernel size
        out_h {[int]} -- output height
        out_w {[int]} -- output width
    
    Returns:
        [int] -- pad_top, pad_down, pad_left, pad_right
    """
    p_h = s*(h-1)+k-out_h
    p_w = s*(w-1)+k-out_w

    p_top = p_h // 2
    p_down = p_h - p_top

    p_left = p_w // 2
    p_right = p_w - p_left 

    return int(p_top), int(p_down), int(p_left), int(p_right)

