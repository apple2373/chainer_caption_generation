'''
To extarct CNN features.

This code could be messy.
I did not assume others  use this, but decided to make avaiable, 
because I saw many people who wants to use VGG insetad of GoogleNet.
But remember that this is for GoogleNet.
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check
import chainer 

import argparse
import os
import numpy as np
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
#import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave
import json
import nltk
import random
import pickle
import math
import skimage.transform


#Settings can be changed by command line arguments
gpu_id=-1# GPU ID. if you want to use cpu, -1
#gpu_id=0
savedir='../work/img_features/'# name of log and results image saving directory
image_feature_dim=1024#特徴の次元数。

#Functions
def get_image_ids(file_place):
    
    f = open(file_place, 'r')
    jsonData = json.load(f)
    f.close()
    
    image_id2feature={}
    for caption_data in jsonData['annotations']:
        image_id=caption_data['image_id']
        image_id2feature[image_id]=np.array([image_feature_dim,])

    return image_id2feature

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#画像読み込み関数
#ただ読むだけ
MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
def image_read_np(file_place):
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
    return rawim.transpose(2, 0, 1).astype(np.float32)

#main

# Prepare dataset
file_place = '../data/MSCOCO/annotations/captions_train2014.json'
train_image_id2feature=get_image_ids(file_place)
file_place = '../data/MSCOCO/annotations/captions_val2014.json'
val_image_id2feature=get_image_ids(file_place)


#Caffeモデルをロード
print "loading caffe models"
func = caffe.CaffeFunction('../data/bvlc_googlenet.caffemodel')
if gpu_id>= 0:
    func.to_gpu()
print "done"



print 'feature_exractor'
file_base='../data/MSCOCO/train2014/COCO_train2014_'
for i, image_id in enumerate(train_image_id2feature.keys()):

    if i%5000==0:
        print i 

    try:
        image=image_read_np(file_base+str("{0:012d}".format(image_id)+'.jpg'))
    except Exception as e:
        print 'image reading error'
        print 'type:' + str(type(e))
        print 'args:' + str(e.args)
        print 'message:' + e.message
        print image_id
        continue

    x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
    x_batch[0]=image
    if gpu_id >=0:
        x = Variable(cuda.to_gpu(x_batch), volatile=True)
    else:
        x = Variable(x_batch, volatile=True)
    image_feature_chainer, = func(inputs={'data': x}, outputs=['pool5/7x7_s1'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                  train=False)

    image_feature_np=image_feature_chainer.data.reshape(1024)
    train_image_id2feature[image_id]=cuda.to_cpu(image_feature_np)


pickle.dump(train_image_id2feature, open(savedir+"train_image_id2feature.pkl", 'wb'), -1)

print "for test"
file_base='../data/MSCOCO/val2014/COCO_val2014_'
for i, image_id in enumerate(val_image_id2feature.keys()):

    if i%5000==0:
        print i 

    try:
        image=image_read_np(file_base+str("{0:012d}".format(image_id)+'.jpg'))
    except Exception as e:
        print 'image reading error'
        print 'type:' + str(type(e))
        print 'args:' + str(e.args)
        print 'message:' + e.message
        print image_id
        continue

    x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
    x_batch[0]=image
    if gpu_id >=0:
        x = Variable(cuda.to_gpu(x_batch), volatile=True)
    else:
        x = Variable(x_batch, volatile=True)
    image_feature_chainer, = func(inputs={'data': x}, outputs=['pool5/7x7_s1'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                  train=False)

    image_feature_np=image_feature_chainer.data.reshape(1024)
    val_image_id2feature[image_id]=cuda.to_cpu(image_feature_np)

pickle.dump(val_image_id2feature, open(savedir+"val_image_id2feature.pkl", 'wb'), -1)