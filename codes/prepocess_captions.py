#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program preprocesses the caption into picke file.
Main purpose is to tokenize, make lower case, and filter out low frequent vocaburaries.
Note tokenize and  make lower case is done by a function (read_MSCOCO_json) in another file MSCOCO.py.
"""

from MSCOCO import read_MSCOCO_json #to read MSCOCO json file. 
from gensim import corpora
import pickle

file_place = '../data/MSCOCO/annotations/captions_train2014.json'
train_captions,train_caption_id2tokens,train_caption_id2image_id = read_MSCOCO_json(file_place)

texts=train_caption_id2tokens.values()
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=1.0)
dictionary.compactify() # remove gaps in id sequence after words that were removed
index2token = dict((v, k) for k, v in dictionary.token2id.iteritems())
ukn_id=len(dictionary.token2id)
index2token[ukn_id]='<UKN>'

#just save the map from index to token (word)
#that means this is vocaburary file
with open('../work/index2token.pkl', 'w') as f:
    pickle.dump(index2token,f)


train_caption_id2sentence={}
for (caption_id,tokens) in train_caption_id2tokens.iteritems():
    sentence=[]
    for token in tokens:
        if token in dictionary.token2id:
            sentence.append(dictionary.token2id[token])
        else:
            sentence.append(ukn_id)
            
    train_caption_id2sentence[caption_id]=sentence


#Save preprocessed captions. 
with open('../work/preprocessed_train_captions.pkl', 'w') as f:
    pickle.dump((train_captions,train_caption_id2sentence,train_caption_id2image_id),f)