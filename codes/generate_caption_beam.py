# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import argparse
from image_reader import Image_reader
from caption_generator import Caption_generator

#Settings can be changed by command line arguments
gpu_id=-1# GPU ID. if you want to use cpu, -1
model_place='../models/caption_model.chainer'
caffe_model_place='../data/bvlc_googlenet_caffe_chainer.pkl'
index2word_file = '../work/index2token.pkl'
image_file_name='../images/test_image.jpg'
beamsize=3

#Override Settings by argument
parser = argparse.ArgumentParser(description=u"caption generation")
parser.add_argument("-g", "--gpu",default=gpu_id, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("-m", "--model",default=model_place, type=str, help=u" caption generation model")
parser.add_argument("-c", "--caffe",default=caffe_model_place, type=str, help=u" pre trained caffe model pickled after imported to chainer")
parser.add_argument("-v", "--vocab",default=index2word_file, type=str, help=u" vocaburary file")
parser.add_argument("-i", "--image",default=image_file_name, type=str, help=u"a image that you want to generate capiton ")
parser.add_argument("-b", "--beam",default=beamsize, type=int, help=u"a image that you want to generate capiton ")

args = parser.parse_args()
gpu_id=args.gpu
model_place= args.model
index2word_file = args.vocab
image_file_name = args.image
caffe_model_place = args.caffe
beamsize = args.beam


#Instantiate image_reader with GoogleNet mean image
mean_image = np.array([104, 117, 123]).reshape((3,1,1))#GoogleNet Mean
image_reader=Image_reader(mean=mean_image)

#Instantiate caption generator
caption_generator=Caption_generator(caption_model_place=model_place,cnn_model_place=caffe_model_place,index2word_place=index2word_file,beamsize=beamsize,gpu_id=gpu_id)

#Read Image
image=image_reader.read(image_file_name)

#Generate Catpion
captions=caption_generator.generate(image)

#print it
for caption in captions:
    sentence=caption['sentence']
    probability=caption['probability']
    print " ".join(sentence),probability

