# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
If you want to integrate caption generation system for your system, you can import this module.
'''

import os
#comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
#os.environ["CHAINER_TYPE_CHECK"] = "0" 
import chainer 
#If the below is false, the type check is disabled. 
#print(chainer.functions.Linear(1,1).type_check_enable) 

import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers
import pickle
import copy
from image_reader import Image_reader

class Caption_generator(object):
    def __init__(self,caption_model_place,cnn_model_place,index2word_place,gpu_id=-1,beamsize=3):
        #basic paramaters you need to modify
        self.gpu_id=gpu_id# GPU ID. if you want to use cpu, -1
        self.beamsize=beamsize

        #Gpu Setting
        global xp
        if self.gpu_id >= 0:
            xp = cuda.cupy 
            cuda.get_device(gpu_id).use()
        else:
            xp=np

        # Prepare dataset
        with open(index2word_place, 'r') as f:
            self.index2word = pickle.load(f)
        vocab=self.index2word

        #Load Caffe Model
        with open(cnn_model_place, 'r') as f:
            self.func = pickle.load(f)

        #Model Preparation
        image_feature_dim=1024#dimension of image feature
        self.n_units = 512  #number of units per layer
        n_units = 512 
        self.model = FunctionSet()
        self.model.img_feature2vec=F.Linear(image_feature_dim, n_units)#CNN(I)の最後のレイヤーに相当。#parameter  W,b
        self.model.embed=F.EmbedID(len(vocab), n_units)#W_e*S_tに相当 #parameter  W
        self.model.l1_x=F.Linear(n_units, 4 * n_units)#parameter  W,b
        self.model.l1_h=F.Linear(n_units, 4 * n_units)#parameter  W,b
        self.model.out=F.Linear(n_units, len(vocab))#parameter  W,b
        serializers.load_hdf5(caption_model_place, self.model)#read pre-trained model

        #To GPU
        if gpu_id >= 0:
            self.model.to_gpu()
            self.func.to_gpu()

        #to avoid overflow.
        #I don't know why, but this model overflows at the first time only with CPU.
        #So I intentionally make overflow so that it never happns after that.
        if gpu_id < 0:
            numpy_image = np.ones((3, 224,224), dtype=np.float32)
            self.generate(numpy_image)

    def feature_exractor(self,x_chainer_variable): #to extract image feature by CNN.
        y, = self.func(inputs={'data': x_chainer_variable}, outputs=['pool5/7x7_s1'],
                      disable=['loss1/ave_pool', 'loss2/ave_pool','loss3/classifier'],
                      train=False)
        return y

    def forward_one_step_for_image(self,img_feature, state, volatile='on'):
        x = img_feature#img_feature is chainer.variable.
        h0 = self.model.img_feature2vec(x)
        h1_in = self.model.l1_x(F.dropout(h0,train=False)) + self.model.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        y = self.model.out(F.dropout(h1,train=False))#don't forget to change drop out into non train mode.
        state = {'c1': c1, 'h1': h1}
        return state, F.softmax(y)

    #forward_one_step is after the CNN layer, 
    #h0 is n_units dimensional vector (embedding)
    def forward_one_step(self,cur_word, state, volatile='on'):
        x = chainer.Variable(cur_word, volatile)
        h0 = self.model.embed(x)
        h1_in = self.model.l1_x(F.dropout(h0,train=False)) + self.model.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        y = self.model.out(F.dropout(h1,train=False)) 
        state = {'c1': c1, 'h1': h1}
        return state, F.softmax(y)

    def beam_search(self,sentence_candidates,final_sentences,depth=1,beamsize=3):
        volatile=True
        next_sentence_candidates_temp=list()
        for sentence_tuple in sentence_candidates:
            cur_sentence=sentence_tuple[0]
            cur_index=sentence_tuple[0][-1]
            cur_index_xp=xp.array([cur_index],dtype=np.int32)
            cur_state=sentence_tuple[1]
            cur_log_likely=sentence_tuple[2]

            state, predicted_word = self.forward_one_step(cur_index_xp,cur_state, volatile=volatile)
            predicted_word_np=cuda.to_cpu(predicted_word.data)
            top_indexes=(-predicted_word_np).argsort()[0][:beamsize]

            for index in np.nditer(top_indexes):
                index=int(index)
                probability=predicted_word_np[0][index]
                next_sentence=copy.deepcopy(cur_sentence)
                next_sentence.append(index)
                log_likely=math.log(probability)
                next_log_likely=cur_log_likely+log_likely
                next_sentence_candidates_temp.append((next_sentence,state,next_log_likely))# make each sentence tuple

        prob_np_array=np.array([sentence_tuple[2] for sentence_tuple in next_sentence_candidates_temp])
        top_candidates_indexes=(-prob_np_array).argsort()[:beamsize]
        next_sentence_candidates=list()
        for i in top_candidates_indexes:
            sentence_tuple=next_sentence_candidates_temp[i]
            index=sentence_tuple[0][-1]
            if self.index2word[index]=='<EOS>':
                final_sentence=sentence_tuple[0]
                final_likely=sentence_tuple[2]
                final_probability=math.exp(final_likely)
                final_sentences.append((final_sentence,final_probability,final_likely))
            else:
                next_sentence_candidates.append(sentence_tuple)

        if len(final_sentences)>=beamsize:
            return final_sentences
        elif depth==50:
            return final_sentences
        else:
            depth+=1
            return self.beam_search(next_sentence_candidates,final_sentences,depth,beamsize)

    def generate(self,numpy_image):
        '''Generate Caption for an Numpy Image array
        
        Args:
            numpy_image: numpy image

        Returns:
            list of generated captions. The structure is [caption,caption,caption,...]
            Where caption = {"sentence":This is a generated sentence, "probability": The probability of the generated sentence} 

        '''

        #initial step
        x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
        x_batch[0]=numpy_image

        volatile=True
        if self.gpu_id  >=0:
            x_batch_chainer = Variable(cuda.to_gpu(x_batch),volatile=volatile)
        else:
            x_batch_chainer = Variable(x_batch,volatile=volatile)

        batchsize=1
        #image is chainer.variable.
        state = {name: chainer.Variable(xp.zeros((batchsize, self.n_units),dtype=np.float32),volatile) for name in ('c1', 'h1')}
        img_feature=self.feature_exractor(x_batch_chainer)
        state, predicted_word = self.forward_one_step_for_image(img_feature,state, volatile=volatile)

        if self.gpu_id >=0:
            index=cuda.to_cpu(predicted_word.data.argmax(1))[0]
        else:
            index=predicted_word.data.argmax(1)[0]

        probability=predicted_word.data[0][index]
        initial_sentence_candidates=[([index],state,probability)]

        final_sentences=list()
        generated_sentence_candidates=self.beam_search(initial_sentence_candidates,final_sentences,beamsize=self.beamsize)

        #convert to index to strings

        generated_string_sentence_candidates=[]
        for sentence_tuple in generated_sentence_candidates:
            sentence=[self.index2word[index] for index in sentence_tuple[0]][1:-1]
            probability=sentence_tuple[1]
            final_likely=sentence_tuple[2]

            a_candidate={'sentence':sentence,'probability':probability,'log_probability':final_likely}
    
            generated_string_sentence_candidates.append(a_candidate)


        return generated_string_sentence_candidates

    def generate_temp(self,numpy_image):

        '''Simple Generate Caption for an Numpy Image array
        
        Args:
            numpy_image: numpy image

        Returns:
            string of generated capiton
        '''

        genrated_sentence_string=''
        x_batch = np.ndarray((1, 3, 224,224), dtype=np.float32)
        x_batch[0]=numpy_image

        volatile=True
        if self.gpu_id >=0:
            x_batch_chainer = Variable(cuda.to_gpu(x_batch),volatile=volatile)
        else:
            x_batch_chainer = Variable(x_batch,volatile=volatile)

        batchsize=1

        #image is chainer.variable.
        state = {name: chainer.Variable(xp.zeros((batchsize, self.n_units),dtype=np.float32),volatile) for name in ('c1', 'h1')}
        img_feature=self.feature_exractor(x_batch_chainer)
        #img_feature_chainer is chainer.variable of extarcted feature.
        state = {name: chainer.Variable(xp.zeros((batchsize, self.n_units),dtype=np.float32),volatile) for name in ('c1', 'h1')}
        state, predicted_word = self.forward_one_step_for_image(img_feature,state, volatile=volatile)
        index=predicted_word.data.argmax(1)
        index=cuda.to_cpu(index)[0]
        #genrated_sentence_string+=index2word[index] #dont's add it because this is <SOS>

        for i in xrange(50):
            state, predicted_word = self.forward_one_step(predicted_word.data.argmax(1).astype(np.int32),state, volatile=volatile)
            index=predicted_word.data.argmax(1)
            index=cuda.to_cpu(index)[0]
            if self.index2word[index]=='<EOS>':
                genrated_sentence_string=genrated_sentence_string.strip()
                break;
            genrated_sentence_string+=self.index2word[index]+" "

        return genrated_sentence_string

    def get_top_sentence(self,numpy_image):
        '''
        just get a top sentence as  string
        
        Args:
            numpy_image: numpy image

        Returns:
            string of generated capiton
        '''
        candidates=self.generate(numpy_image)
        scores=[caption['log_probability'] for caption in candidates]
        argmax=np.argmax(scores)
        top_caption=candidates[argmax]['sentence']

        sentence = ''
        for word in top_caption:
            sentence+=word+' '

        return sentence.strip()



