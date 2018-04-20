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