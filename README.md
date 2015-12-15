#image cpation generation by chainer
This codes are trying to reproduce the image captioning by google in CVPR 2015.
Show and Tell: A Neural Image Caption Generator
http://arxiv.org/abs/1411.4555

The training data is MSCOCO. I used GoogleNet to extract  images feature in addance (preprosessing), and then traind langauge model to generate caption.  

More information is in my blog post. 

##requirement
chainer 1.5  
and some more packages.   
If you are new, I suggest you to install Anaconda and then install chainer.  

##I just want to generate caption!
OK, first, save the all the folders. then, go to the codes2 folder.
You can run 
```
cd codes
python generate_caption.py -i ../images/test_image.jpg
```
This generate a caption for ../images/test_image.jpg. If you want to use your image, you just have to indicate -i option to image that you want to generate captions. 

##I want to the model by myself.
MSCOCO caption data is preprocessed and pickled, so you can train easily.  
```
 cd codes
 python train_caption_model.py 
 python train_caption_model.py  -g 0 # to use gpu. change the number to gpu_id
```
The log and trained model will be saved to a directory (experiment1 is defalt)  
If you want to change, use -d option. 
```
 python train_caption_model.py -d ./yourdirectory
```


##I wnat to train other data.
Sorry, current implementation does not suport it. You need to preprocess the data. Maybe you can read and modify the code. 

##I wnat to fine-tune CNN part. 
Sorry, current implementation does not suport it. Maybe you can read and modify the code. 
