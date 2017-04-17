### I no longer maintain this repository. This implementation is not that clean and hard to use if you want to train on your own data. I re-implemented from scratch. The new one is much faster, accurate, and clean. It can even generate Chinese captions. Please see the [better implementation] (https://github.com/apple2373/chainer-caption).


# image caption generation by chainer
This codes are trying to reproduce the image captioning by google in CVPR 2015.
Show and Tell: A Neural Image Caption Generator
http://arxiv.org/abs/1411.4555

The training data is MSCOCO. I used GoogleNet to extract  images feature in advance (preprocessed them before training), and then trained language model to generate caption.

I made pre-trained model available. The model achieves CIDEr of 0.66 for the MSCOCO validation dataset. To achieve the better score, the use of beam search is first step (not implemented yet). Also, I think the CNN has to be fine-tuned.  
Update: I implemented a beam search. Check the usage below.  

More information including some sample captions are in my blog post. 
http://t-satoshi.blogspot.com/2015/12/image-caption-generation-by-cnn-and-lstm.html

## requirement
chainer 1.6  http://chainer.org
and some more packages.  
!!Warning ** Be sure to use chainer 1.6.**  Not the latest version. If you have another version, no guarantee to work.  
If you are new, I suggest you to install Anaconda (https://www.continuum.io/downloads) and then install chainer.  You can watch the video below. 

## I have a problem to prepare environment
I  prepared a video to show how you prepare environment and generate captions on ubuntu. I used a virtual machine just after installing ubuntu 14.04. If you imitate as in the video, you can generate captions. The process is almost the same for Mac. Windows is not suported because I cannot use it (Acutually chainer does not officialy support windows). 
https://drive.google.com/file/d/0B046sNk0DhCDUkpwblZPME1vQzg/edit
Or, some commands that might help:
```
#get and install anaconda. you might want to check the latest link.
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh
bash Anaconda2-2.4.1-Linux-x86_64.sh -b
echo 'export PATH=$HOME/anaconda/bin:$PATH' >> .bashrc
echo 'export PYTHONPATH=$HOME/anaconda/lib/python2.7/site-packages:$PYTHONPATH' >> .bashrc
source .bashrc
conda update conda -y
# install chainer 
pip install chainer==1.6
```

## I just want to generate caption!
OK, first, you need to download the models and other preprocessed files.
Then you can generate caption.

IMPORTANT NOTE:  
Google Drive suddenly shut down the hosting service and the file downlaod no longer works.  
Ref: https://gsuiteupdates.googleblog.com/2015/08/deprecating-web-hosting-support-in.html

I don't have time to uplaod somewhere else, but all files are here:  
https://drive.google.com/open?id=0B046sNk0DhCDeEczcm1vaWlCTFk  

```
bash download.sh
cd codes
python generate_caption.py -i ../images/test_image.jpg
```
This generate a caption for ../images/test_image.jpg. If you want to use your image, you just have to indicate -i option to image that you want to generate captions. 

Once you set up environment, you can use it as a module.Check the ipython notebooks. This includes beam search. 
English:https://github.com/apple2373/chainer_caption_generation/blob/master/codes/sample_code.ipynb  

Also, you can try beam search as:
```
cd codes
python generate_caption_beam.py -b 3 -i ../images/test_image.jpg
```
-b option indicates beam size. Default is 3. 

## I want to train the model by myself.
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

## I want to train from other data.
Sorry, current implementation does not support it. You need to preprocess the data. Maybe you can read and modify the code. 

## I want to fine-tune CNN part. 
Sorry, current implementation does not support it. Maybe you can read and modify the code. 

## I want to generate Japanese caption. 
I made pre-trained Japanese caption model available.  You can download Japanese caption model with the following script.
```
bash download.sh 
bash download_jp.sh
```
```
cd codes
python generate_caption.py -v ../work/index2token_jp.pkl -m ../models/caption_model_jp.chainer -i ../images/test_image.jpg
```
Japnese Notebook: https://github.com/apple2373/chainer_caption_generation/blob/master/codes/sample_code_jp.ipynb  
Japnese Blogpost: http://t-satoshi.blogspot.com/2016/01/blog-post_1.html  
