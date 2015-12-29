#! /bin/bash
cd work
if [ ! -f index2token_jp.pkl ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/work/index2token_jp.pkl
fi
cd ..
cd models
if [ ! -f caption_model_jp.chainer ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDeEczcm1vaWlCTFk/models/caption_model_jp.chainer
fi
