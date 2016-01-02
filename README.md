#image caption generation by chainer
This codes are trying to reproduce the image captioning by google in CVPR 2015.
Show and Tell: A Neural Image Caption Generator
http://arxiv.org/abs/1411.4555

The training data is MSCOCO. I used GoogleNet to extract  images feature in advance (preprocessed them before training), and then trained language model to generate caption.

I made pre-trained model available. The model achieves CIDEr of 0.66 for the MSCOCO validation dataset. To achieve the better score, the use of beam search is first step (not implemented yet). Also, I think the CNN has to be fine-tuned.

More information including some sample captions are in my blog post. 
http://t-satoshi.blogspot.com/2015/12/image-caption-generation-by-cnn-and-lstm.html

##requirement
chainer 1.5  http://chainer.org
and some more packages.  
If you are new, I suggest you to install Anaconda and then install chainer.  You can watch the video below. 

##I have a problem to prepare environment
I  prepared a video to show how you prepare environment and generate captions on ubuntu. I used a virtual machine just after installing ubuntu 14.04. If you imitate as in the video, you can generate captions. The process is almost the same for Mac. Windows is not suported because I cannot use it (Acutually chsiner does not officialy support windows). 
https://drive.google.com/file/d/0B046sNk0DhCDUkpwblZPME1vQzg/edit

##I just want to generate caption!
OK, first, you need to download the models and other preprocessed files.
Then you can generate caption.
```
bash download.sh
cd codes
python generate_caption.py -i ../images/test_image.jpg
```
This generate a caption for ../images/test_image.jpg. If you want to use your image, you just have to indicate -i option to image that you want to generate captions. 

Once you set up environment, you can use it as a module.Check the ipy notebooks. 
English:https://github.com/apple2373/chainer_caption_generation/blob/master/codes/sample_code.ipynb  
Japnese: https://github.com/apple2373/chainer_caption_generation/blob/master/codes/sample_code_jp.ipynb

##I want to train the model by myself.
I extracted the GoogleNet features and pickled, so you use it for training.  
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

##I want to train from other data.
Sorry, current implementation does not support it. You need to preprocess the data. Maybe you can read and modify the code. 

##I want to fine-tune CNN part. 
Sorry, current implementation does not support it. Maybe you can read and modify the code. 

##I want to generate Japanese caption. 
I made pre-trained Japanese caption model available.  You can download Japanese caption model with the following script.
```
bash download.sh 
bash download_jp.sh
```
```
cd codes
python generate_caption.py -v ../work/index2token_jp.pkl -m ../models/caption_model_jp.chainer -i ../images/test_image.jpg
```
Japnese Samples: http://t-satoshi.blogspot.com/2016/01/blog-post_1.html