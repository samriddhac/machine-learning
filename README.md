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