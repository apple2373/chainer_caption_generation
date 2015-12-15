'''
This is a script to generate captions for validiation files.
'''

# -*- coding: utf-8 -*-
#!/usr/bin/env python
#compatible chiner 1.5


import os
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 
#Check che below is False if you disabled type check
#print(chainer.functions.Linear(1,1).type_check_enable) 

import argparse
import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers
import pickle

import glob
import os
import json

#Settings can be changed by command line arguments
gpu_id=0# GPU ID. if you want to use cpu, -1
model_dir='../experiment1'

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-m", "--modeldir",default=model_dir, type=str, help=u"The directory that have models")
args = parser.parse_args()
gpu_id=args.gpu
model_dir= args.modeldir


print('pareparing evaluation')


with open('../work/img_features/val_image_id2feature.pkl', 'r') as f:
    val_image_id2feature = pickle.load(f)

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Basic Setting
image_feature_dim=1024#dimension of image feature
n_units = 512  #number of units per layer
batchsize=1#has to be 1 currently because of implementation.
volatile=False


# Prepare dataset
print "loading vocab"
with open('../work/index2token.pkl', 'r') as f:
    index2word = pickle.load(f)

vocab=index2word

#Model Preparation
print "preparing caption generation models"
model = FunctionSet()
model.img_feature2vec=F.Linear(image_feature_dim, n_units)#CNN(I)の最後のレイヤーに相当。#parameter  W,b
model.embed=F.EmbedID(len(vocab), n_units)#W_e*S_tに相当 #parameter  W
model.l1_x=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.l1_h=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.out=F.Linear(n_units, len(vocab))#parameter  W,b

#To GPU
if gpu_id >= 0:
    model.to_gpu()
print "done"

for (image_id,feature) in val_image_id2feature.iteritems():
    x_batch = np.ndarray((1,image_feature_dim), dtype=np.float32)
    x_batch[0]=feature
    if gpu_id >= 0:
        x_batch=cuda.to_gpu(x_batch)
    x_batch_chainer = Variable(x_batch,volatile=volatile)
    val_image_id2feature[image_id]=x_batch_chainer

#Define Newtowork (Forward)

#forward_one_step is after the CNN layer, 
#h0 is n_units dimensional vector (embedding)
def forward_one_step(cur_word, state, volatile=True):
    x = chainer.Variable(cur_word, volatile)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0,train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1,train=False)) 
    state = {'c1': c1, 'h1': h1}
    return state, y

def forward_one_step_for_image(img_feature, state, volatile=True):
    x = img_feature#img_feature is chainer.variable.
    h0 = model.img_feature2vec(x)
    h1_in = model.l1_x(F.dropout(h0,train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1,train=False))#don't forget to change drop out into non train mode.
    state = {'c1': c1, 'h1': h1}
    return state, y

print('evaluation started')

for model_place in glob.glob(os.path.join(model_dir, 'caption_model*.chainer')):
    print model_place

    serializers.load_hdf5(model_place, model)#load model

    results_list=[]

    for image_id in val_image_id2feature:

        img_feature_chainer=val_image_id2feature[image_id]

        genrated_sentence_string=''

        #img_feature_chainer is chainer.variable of extarcted feature.
        state = {name: chainer.Variable(xp.zeros((batchsize, n_units),dtype=np.float32),volatile) for name in ('c1', 'h1')}
        state, predicted_word = forward_one_step_for_image(img_feature_chainer,state, volatile=volatile)
        index=predicted_word.data.argmax(1)
        index=cuda.to_cpu(index)[0]
        #genrated_sentence_string+=index2word[index] #dont's add it because this is <SOS>

        for i in xrange(50):
            state, predicted_word = forward_one_step(predicted_word.data.argmax(1).astype(np.int32),state, volatile=volatile)
            index=predicted_word.data.argmax(1)
            index=cuda.to_cpu(index)[0]
            if index2word[index]=='<EOS>':
                genrated_sentence_string=genrated_sentence_string.strip()
                break;
            genrated_sentence_string+=index2word[index]+" "

        line={}
        line['image_id']=image_id
        line['caption']=genrated_sentence_string
        results_list.append(line)
        
    name, ext = os.path.splitext(model_place)
    with open(name+'.json', 'w') as f:
        json.dump(results_list, f, sort_keys=True, indent=4)
