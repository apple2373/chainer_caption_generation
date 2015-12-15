#! /bin/bash
cd data
if [ ! -f bvlc_googlenet_caffe_chainer.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/data/bvlc_googlenet_caffe_chainer.pkl
fi
cd ..
cd work
if [ ! -f index2token.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/work/index2token.pkl
fi
if [ ! -f preprocessed_train_captions.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/work/preprocessed_train_captions.pkl
fi
if [ ! -d img_features ]; then
	mkdir img_features
fi
cd img_features
if [ ! -f train_image_id2feature.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/work/img_features/train_image_id2feature.pkl
fi
if [ ! -f val_image_id2feature.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/work/img_features/val_image_id2feature.pkl
fi
cd ../../
cd models
if [ ! -f caption_model.chainer ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/models/caption_model.chainer
fi