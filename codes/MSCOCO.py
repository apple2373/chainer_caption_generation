#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import nltk 

def read_MSCOCO_json(file_place):
        
    f = open(file_place, 'r')
    jsonData = json.load(f)
    f.close()

    captions={}#key is sentence_length. 
    caption_id2tokens={}
    caption_id2image_id={}

    for caption_data in jsonData['annotations']:
        caption_id=caption_data['id']
        image_id=caption_data['image_id']
        caption=caption_data['caption']

        caption=caption.replace('\n', '').strip().lower()
        if caption[-1]=='.':#to delete the last period. 
            caption=caption[0:-1]

        caption_tokens=['<SOS>']
        caption_tokens += nltk.word_tokenize(caption)
        caption_tokens.append("<EOS>")
        caption_length=len(caption_tokens)

        if caption_length in captions:
            captions[caption_length].add(caption_id)
        else:
            captions[caption_length]=set([caption_id])
        
        caption_id2tokens[caption_id]=caption_tokens
        caption_id2image_id[caption_id]=image_id
        
    return captions,caption_id2tokens,caption_id2image_id
       