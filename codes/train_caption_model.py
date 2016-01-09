#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 
#Check che below is False if you disabled type check
#print(chainer.functions.Linear(1,1).type_check_enable) 

import argparse
import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers
import pickle
import random

#Settings can be changed by command line arguments
gpu_id=-1# GPU ID. if you want to use cpu, -1
#gpu_id=4
savedir='../experiment1/'# name of log and results image saving directory

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-d", "--savedir",default=savedir, type=str, help=u"The directory to save models and log")
args = parser.parse_args()
gpu_id=args.gpu
savedir=args.savedir

#Gpu Setting
if gpu_id >= 0:
    xp = cuda.cupy 
    cuda.get_device(gpu_id).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed data")

with open('../work/index2token.pkl', 'r') as f:
    index2token = pickle.load(f)

with open('../work/preprocessed_train_captions.pkl', 'r') as f:
    train_captions,train_caption_id2sentence,train_caption_id2image_id = pickle.load(f)

with open('../work/img_features/train_image_id2feature.pkl', 'r') as f:
    train_image_id2feature = pickle.load(f)

#Model Preparation
print "preparing caption generation models"
image_feature_dim=1024#特徴の次元数。
n_units = 512  # number of units per layer
vocab_size=len(index2token)

model = chainer.FunctionSet()
model.img_feature2vec=F.Linear(image_feature_dim, n_units)#CNN(I)の最後のレイヤーに相当。#parameter  W,b
model.embed=F.EmbedID(vocab_size, n_units)#W_e*S_tに相当 #parameter  W
model.l1_x=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.l1_h=F.Linear(n_units, 4 * n_units)#parameter  W,b
model.out=F.Linear(n_units, vocab_size)#parameter  W,b

#Parameter Initialization
#Mimicked Chainer Samples
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

#set forget bias 1
model.l1_x.b.data[2*n_units:3*n_units]=np.ones(model.l1_x.b.data[2*n_units:3*n_units].shape).astype(xp.float32)
model.l1_h.b.data[2*n_units:3*n_units]=np.ones(model.l1_h.b.data[2*n_units:3*n_units].shape).astype(xp.float32)

#To GPU
if gpu_id >= 0:
    model.to_gpu()


#Define Newtowork (Forward)

#forward_one_stepは画像の話は無視。それはforwardの一回目で特別にやる。
#h0はn_units次元のベクトル(embedding)
#cur_wordはその時の単語のone-hot-vector
#next_wordはそこで出力すべきone-hot-vector(つまり次のー単語)


def forward_one_step(cur_word, next_word, state, volatile=False):
    x = chainer.Variable(cur_word, volatile)
    t = chainer.Variable(next_word, volatile)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1)) 
    state = {'c1': c1, 'h1': h1}
    loss = F.softmax_cross_entropy(y, t)
    return state, loss

def forward_one_step_for_image(img_feature, first_word, state, volatile=False):
    print img_feature.shape
    x = chainer.Variable(img_feature)
    t = chainer.Variable(first_word, volatile)
    h0 = model.img_feature2vec(x)
    h1_in = model.l1_x(F.dropout(h0)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    y = model.out(F.dropout(h1))
    state = {'c1': c1, 'h1': h1}
    loss = F.softmax_cross_entropy(y, t)
    return state, loss

#imageは画像
#x_listはある画像(image)に対応する文章（単語の集まり+EOS）
#つまりx_list=[word1,word2,....,EOS]
def forward(img_feature,sentences, volatile=False):
    #imageはすでにchinaer variableである。
    state = {name: chainer.Variable(xp.zeros((batchsize, n_units),dtype=xp.float32),volatile) for name in ('c1', 'h1')}
    loss = 0
            
    first_word=sentences.T[0]
    #[[w11,w12,...],[w21,w22...]]から[w11,w21]と最初の単語たちを取り出す.
    #バッチサイズの数だけ文があって、それぞれの最初の単語だけを取ってきた、一次元の配列を作るということ。

    state, new_loss = forward_one_step_for_image(img_feature, first_word,state, volatile=volatile)
    loss += new_loss
    
    #cur_wordに今の単語のnp.array(1次元)
    #next_wordに次の単語のnp.array(1次元)
    for cur_word, next_word in zip(sentences.T, sentences.T[1:]):
        state, new_loss = forward_one_step(cur_word, next_word,state, volatile=volatile)
        loss += new_loss
    return loss

optimizer = optimizers.Adam()
optimizer.setup(model)

#Trining Setting
normal_batchsize=256
grad_clip = 1.0
num_train_data=len(train_caption_id2image_id)

#Begin Training
print 'training started'
for epoch in xrange(200):

    print 'epoch %d' %epoch

    batchsize=normal_batchsize
    caption_ids_batches=[]
    for caption_length in train_captions.keys():
        caption_ids_set=train_captions[caption_length]
        caption_ids=list(caption_ids_set)
        random.shuffle(caption_ids)
        caption_ids_batches+=[caption_ids[x:x + batchsize] for x in xrange(0, len(caption_ids), batchsize)]   
    random.shuffle(caption_ids_batches)

    # training_bacthes={}
    # for i, caption_ids_batch in enumerate(caption_ids_batches):
    #     images = xp.array([train_image_id2feature[train_caption_id2image_id[caption_id]] for caption_id in caption_ids_batch],dtype=xp.float32)
    #     sentences = xp.array([train_caption_id2sentence[caption_id] for caption_id in caption_ids_batch],dtype=xp.int32)
    #     training_bacthes[i]= (images,sentences)

    #This is equivalent for above and hard to read, but I inteitionally did for faster calculation
    training_bacthes = \
        { i:\
            (\
                xp.array([train_image_id2feature[train_caption_id2image_id[caption_id]] for caption_id in caption_ids_batch],dtype=xp.float32),\
                xp.array([train_caption_id2sentence[caption_id] for caption_id in caption_ids_batch],dtype=xp.int32)\
            )\
        for i, caption_ids_batch in enumerate(caption_ids_batches)\
        }

    sum_loss = 0
    for i, batch in training_bacthes.iteritems():
        images=batch[0]
        sentences=batch[1]

        sentence_length=len(sentences[0])
        batchsize=normal_batchsize#reverse batchsize if it is changed due to sentence length.
        if len(images) != batchsize:
            batchsize=len(images) 
            #last batch may be less than batchsize. Or depend on caption_length

        optimizer.zero_grads()
        loss = forward(images,sentences)
        print loss.data
        with open(savedir+"real_loss.txt", "a") as f:
            f.write(str(loss.data)+'\n') 
        with open(savedir+"real_loss_per_word.txt", "a") as f:
            f.write(str(loss.data/sentence_length)+'\n') 

        loss.backward()
        #optimizer.clip_grads(grad_clip)
        optimizer.update()
        
        sum_loss      += loss.data * batchsize
    
    serializers.save_hdf5(savedir+"/caption_model"+str(epoch)+'.chainer', model)
    serializers.save_hdf5(savedir+"/optimizer"+str(epoch)+'.chainer', optimizer)

    mean_loss     = sum_loss / num_train_data
    with open(savedir+"mean_loss.txt", "a") as f:
        f.write(str(loss.data)+'\n')

