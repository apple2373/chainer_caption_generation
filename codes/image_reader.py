#!/usr/bin/env python
# -*- coding: utf-8 -*-

#under construction! 
#curerntly, I do not use it. So you can ignore.



from scipy.misc import imread, imresize
import skimage.transform

class image_reader(object):
    def __init__():
        self.mean_image = np.zeros((3,1,1))


    #画像読み込み関数
    #ただ読むだけ
    #taken from https://github.com/ebenolson/Recipes/blob/master/examples/imagecaption/COCO%20Preprocessing.ipynb
    #see also https://groups.google.com/forum/#!topic/lasagne-users/cCFVeT5rw-o
    def read_np(file_place):
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

        im = im - mean_image
        return rawim.transpose(2, 0, 1).astype(np.float32)