# -*- coding: utf-8 -*-
#!/usr/bin/env python
#compatible chiner 1.5


import os
#comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
#os.environ["CHAINER_TYPE_CHECK"] = "0" 
import chainer 
#If the below is false, the type check is disabled. 
#print(chainer.functions.Linear(1,1).type_check_enable) 

import argparse
import os
import numpy as np
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
#import matplotlib.pyplot as plt
from chainer import serializers

from scipy.misc import imread, imresize, imsave
import json
import random
import pickle
import math
import skimage.transform

#Settings can be changed by command line arguments
gpu_id=-1# GPU ID. if you want to use cpu, -1
model_place='../models/caption_model.chainer'
caffe_model_place='../data/bvlc_googlenet_caffe_chainer.pkl'
index2word_file = '../work/index2token.pkl'
image_file_name='../images/test_image.jpg'



#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-m", "--model",default=model_place, type=str, help=u" caption generation model")
parser.add_argument("-c", "--caffe",default=caffe_model_place, type=str, help=u" pre trained caffe model pickled after imported to chainer")
parser.add_argument("-v", "--vocab",default=index2word_file, type=str, help=u" vocaburary file")
parser.add_argument("-i", "--image",default=image_file_name, type=str, help=u"a image that you want to generate capiton ")

args = parser.parse_args()
gpu_id=args.gpu
model_place= args.model
index2word_file = args.vocab
image_file_name = args.image
caffe_model_place = args.caffe

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Basic Setting
image_feature_dim=1024#dimension of image feature
n_units = 512  #number of units per layer


# Prepare dataset
print "loading vocab"
with open(index2word_file, 'r') as f:
    index2word = pickle.load(f)

vocab=index2word


#Load Caffe Model
print "loading caffe models"
with open(caffe_model_place, 'r') as f:
    func = pickle.load(f)

if gpu_id>= 0:
    func.to_gpu()
print "done"

def feature_exractor(x_chainer_variable): #to extract image feature by CNN.
    y, = func(inputs={'data': x_chainer_variable}, outputs=['pool5/7x7_s1'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                  train=False)
    return y

#Read image from file into numpy.
#several prosessing is copied from here: 
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

#Model Preparation
print "preparing caption generation models"
model = FunctionSet()
model.img_feature2vec=F.Linear(image_feature_dim, n_units)#CNN(I)の最後のレイヤーに相当。#parameter  W,b
model.embed=F.EmbedID(len(vocab), n_units)#W_e*S_tに相当 #parameter  W
model.l1_x=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.l1_h=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.out=F.Linear(n_units, len(vocab))#parameter  W,b

serializers.load_hdf5(model_place, model)

#To GPU
if gpu_id >= 0:
    model.to_gpu()
print "done"

#Define Newtowork (Forward)

#forward_one_step is after the CNN layer, 
#h0 is n_units dimensional vector (embedding)
def forward_one_step(cur_word, state, volatile='on'):
    x = chainer.Variable(cur_word, volatile)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0,train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1,train=False)) 
    state = {'c1': c1, 'h1': h1}
    return state, y

def forward_one_step_for_image(img_feature, state, volatile='on'):
    x = img_feature#img_feature is chainer.variable.
    h0 = model.img_feature2vec(x)
    h1_in = model.l1_x(F.dropout(h0,train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1,train=False))#don't forget to change drop out into non train mode.
    state = {'c1': c1, 'h1': h1}
    return state, y

#to avoid overflow.
#I don't know why, but this model overflows only at the first time.
#So I intentionally make overflow so that it never happns after that.
if gpu_id < 0:
    x_batch = np.ones((1, 3, 224,224), dtype=np.float32)
    x_batch_chainer = Variable(x_batch)
    img_feature=feature_exractor(x_batch_chainer)
    state = {name: chainer.Variable(xp.zeros((1, n_units),dtype=np.float32)) for name in ('c1', 'h1')}
    state, predicted_word = forward_one_step_for_image(img_feature,state)

def caption_generate(image_file_name):
    print('sentence generation started')

    genrated_sentence=[]
    volatile=True

    image=image_read_np(image_file_name)
    x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
    x_batch[0]=image

    if gpu_id >=0:
        x_batch_chainer = Variable(cuda.to_gpu(x_batch),volatile=volatile)
    else:
        x_batch_chainer = Variable(x_batch,volatile=volatile)

    batchsize=1

    #image is chainer.variable.
    state = {name: chainer.Variable(xp.zeros((batchsize, n_units),dtype=np.float32),volatile) for name in ('c1', 'h1')}
    img_feature=feature_exractor(x_batch_chainer)
    state, predicted_word = forward_one_step_for_image(img_feature,state, volatile=volatile)
    genrated_sentence.append(predicted_word.data)

    for i in xrange(50):
        state, predicted_word = forward_one_step(predicted_word.data.argmax(1).astype(np.int32),state, volatile=volatile)
        genrated_sentence.append(predicted_word.data)

    print("---genrated_sentence--")

    for predicted_word in genrated_sentence:
        if gpu_id >=0:
            index=cuda.to_cpu(predicted_word.argmax(1))[0]
        else:
            index=predicted_word.argmax(1)[0]
        print index2word[index]
        if index2word[index]=='<EOS>':
            xp.max(predicted_word)
            x_batch_chainer = Variable(predicted_word,volatile=volatile)
            print xp.max(F.softmax(x_batch_chainer).data)
            break

caption_generate(image_file_name)