#!/bin/bash

#nvidia installation.
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl -y
curl -O https://s3-us-west-2.amazonaws.com/sam-tensorflow-installers/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get install cuda-9-0 -y
wget https://s3-us-west-2.amazonaws.com/sam-tensorflow-installers/cudnn-9.0-linux-x64-v7.1.tgz
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*


#python
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
wget https://s3-us-west-2.amazonaws.com/sam-tensorflow-installers/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl

#code
git clone https://github.com/samriddhac/machine-learning.git