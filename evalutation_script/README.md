# Evaluation Script for MSCOCO
This code is based on the the follwoing repository.
https://github.com/tylin/coco-caption
To use the scripts here, please copy the three folders and thier contents to this place.
annotations
pycocoevalcap
pycocotools


## How to do evaluation?
Prepare the directory that contains several json files for evaluation.
The json file should be: 
[{"image_id": 404464, "caption": "black and white photo of a man standing in front of a building"}, {"image_id": 380932, "caption": "group of people are on the side of a snowy field"},...]
Then, it will save json file into results folder by the file name. 
