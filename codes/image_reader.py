#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
The class to read an image as numpy array.
This is particurary designed for ImageNet related task.
So, whatever size the input image have, the output will be centor-croped image of 224*224
Also, you can specify the mean image for CNNs like GoogleNet or VGG 
'''

import numpy as np
from scipy.misc import imread, imresize
import skimage.transform

class Image_reader(object):
    def __init__(self,mean=np.zeros((3,1,1))):
        self.mean_image = mean

    #taken from https://github.com/ebenolson/Recipes/blob/master/examples/imagecaption/COCO%20Preprocessing.ipynb
    #see also https://groups.google.com/forum/#!toself.pic/lasagne-users/cCFVeT5rw-o
    def read(self,file_place):
        im = imread(file_place)
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)

        # Resize so smallest dim = 224, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]
        
        rawim = np.copy(im).astype('uint8')
        
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        
        # Convert to BGR
        # We should know OpenCV's default is BGR instead of RGB
        im = im[::-1, :, :]

        im = im - self.mean_image
        return rawim.transpose(2, 0, 1).astype(np.float32)

    def crop_for_plot(self,file_place):
        im = imread(file_place)
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)
        # Resize so smallest dim = 224, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]
        
        rawim = np.copy(im).astype('uint8')
        
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        
        # Convert to BGR
        im = im[::-1, :, :]

        im = im - MEAN_VALUES
        return rawim