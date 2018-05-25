"# machine-learning" 

bootstrap

sudo yum update -y
mkdir -p python/installables
cd python/installables
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
source .bashrc


https://hackernoon.com/aws-ec2-part-4-starting-a-jupyter-ipython-notebook-server-on-aws-549d87a55ba9
nohup python -u .py > cmd.log &

Tensorflow performance improvement tips :-
https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
https://github.com/lakshayg/tensorflow-build


--------------------------------------------------------------------------
Ubuntu
--------------------------------------------------------------------------
sudo apt-get update -y

mkdir -p installables
cd installables

------------------
Install Bazel

sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python -y
wget https://github.com/bazelbuild/bazel/releases/download/0.12.0/bazel-0.12.0-installer-linux-x86_64.sh
chmod +x bazel-0.12.0-installer-linux-x86_64.sh
bash bazel-0.12.0-installer-linux-x86_64.sh --user
cd ../
vi .bashrc
ADD export PATH="$PATH:$HOME/bin"
source .bashrc

--------------------
Install Conda

cd installables
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
cd ../
source .bashrc

---------------------
Build & Install CPU optimized Tensorflow

cd installables
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel -y
conda -V
conda update conda
conda search "^python$"
conda create -n tensorflow_optimized python=3.6.5 anaconda
source activate tensorflow_optimized
pip install --upgrade pip
pip install six numpy wheel packaging appdirs
bazel build -c opt --verbose_failures --copt=-mavx --copt=-msse4.1 --copt=-msse4.2  -k //tensorflow/tools/pip_package:build_pip_package



-----------------------------------------------------------------------
git clone https://github.com/samriddhac/machine-learning.git
conda install numpy pandas matplotlib keras
source deactivate

------------------


Tensorflow installation with GPU support

 
sudo apt-get update -y

mkdir -p installables
cd installables
sudo vi /etc/modprobe.d/nouveau
/**Content**/
blacklist nouveau
options nouveau modeset=0
sudo reboot
cd installables
sudo apt install awscli
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl -y
curl -O https://s3-us-west-2.amazonaws.com/sam-tensorflow-installers/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update -y
sudo apt-get install cuda-9-0 -y
nvidia-smi

Thu May 24 10:13:00 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID K520           Off  | 00000000:00:03.0 Off |                  N/A |
| N/A   28C    P0    40W / 125W |      0MiB /  4037MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

wget https://s3-us-west-2.amazonaws.com/sam-tensorflow-installers/cudnn-9.0-linux-x64-v7.1.tgz
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

.bashrc

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda

source ~/.bashrc


sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python -y
wget https://github.com/bazelbuild/bazel/releases/download/0.12.0/bazel-0.12.0-installer-linux-x86_64.sh
chmod +x bazel-0.12.0-installer-linux-x86_64.sh
bash bazel-0.12.0-installer-linux-x86_64.sh --user
cd ../
vi .bashrc
ADD export PATH="$PATH:$HOME/bin"
source .bashrc

cd installables
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
cd ../
source .bashrc
cd installables
conda create -n tensorflow_optimized python=3.6.5 anaconda
source activate tensorflow_optimized
pip install --upgrade pip

sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel -y
cd installables
git clone https://github.com/tensorflow/tensorflow
cd tensorflow/
./configure

output :-

You have bazel 0.12.0 installed.
Please specify the location of python. [Default is /home/ubuntu/anaconda3/bin/python]:


Found possible Python library paths:
  /home/ubuntu/anaconda3/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/ubuntu/anaconda3/lib/python3.6/site-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n
No jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: Y
Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: N
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: N
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: N
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.0


Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.1


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Do you wish to build TensorFlow with TensorRT support? [y/N]: N
No TensorRT support will be enabled for TensorFlow.

Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]:


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.0]3.0


Do you want to use clang as CUDA compiler? [y/N]: N
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]: N
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
Configuration finished

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl


>>> print(device_lib.list_local_devices())
2018-05-25 09:23:34.007734: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-05-25 09:23:34.008044: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1378] Found device 0 with properties:
name: GRID K520 major: 3 minor: 0 memoryClockRate(GHz): 0.797
pciBusID: 0000:00:03.0
totalMemory: 3.94GiB freeMemory: 3.90GiB
2018-05-25 09:23:34.008079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1457] Adding visible gpu devices: 0
2018-05-25 09:23:34.305479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:938] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-25 09:23:34.305537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:944]      0
2018-05-25 09:23:34.305550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:957] 0:   N
2018-05-25 09:23:34.305750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1070] Created TensorFlow device (/device:GPU:0 with 3650 MB memory) -> physical GPU (device: 0, name: GRID K520, pci bus id: 0000:00:03.0, compute capability: 3.0)
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 2914931548764562267
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 3827302400
locality {
  bus_id: 1
  links {
  }
}
incarnation: 7690126783017632181
physical_device_desc: "device: 0, name: GRID K520, pci bus id: 0000:00:03.0, compute capability: 3.0"
]









