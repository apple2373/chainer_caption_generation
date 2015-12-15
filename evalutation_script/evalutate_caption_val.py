'''
This is a script to evaluate generated captions for validiation files.
Most of the script are from https://github.com/tylin/coco-caption
'''

# -*- coding: utf-8 -*-
#!/usr/bin/env python
#compatible chiner 1.5

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

model_dir='../experiment1'

annFile='./annotations/captions_val2014.json'

# create coco object and cocoRes object
coco = COCO(annFile)

all_results_json=[]

for i in xrange(50):
    resFile=model_dir+'/caption_model%d.json'%i
    print resFile


    cocoRes = coco.loadRes(resFile)
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    #cocoEval.params['image_id'] = cocoRes.getImgIds()

    #evaluate results
    cocoEval.evaluate()

    # print output evaluation scores
    results={}
    for metric, score in cocoEval.eval.items():
        results[metric]=score
    all_results_json.append(results)

with open(model_dir+'/evaluation_val.json', 'w') as f:
    json.dump(all_results_json, f, sort_keys=True, indent=4)
