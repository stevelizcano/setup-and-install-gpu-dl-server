# This script is designed to work with ubuntu 16.04 LTS
# This is a heavily modified fork of the fast.ai install-gpu server.
# This will install the following:
# nvidia cuda drivers
# Anaconda 3, tensorflow, pytorch, theano and keras 2.0 environment within conda, 
# Python 2.7 environment, with theano and keras 1.2.2
# within each environment it will install advanced packages for common deep-learning utilities
# Keras will be setup for tensorflow, NOT theano
# Jupyter notebook, configured for a server
# Installs Facets, the Google data visualization program

# ###########
# MODIFIED FOR PYTHON 3.6, TENSORFLOW, KERAS 2, PYTORCH, AND SERVER FACING JUPYTER NOTEBOOK
# ###########


# ensure system is updated and has basic build tools
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common



# download and install GPU drivers
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" -O "cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"

sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo modprobe nvidia
nvidia-smi

# install Anaconda for current user
mkdir downloads
cd downloads
wget "https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh" -O "Anaconda3-4.3.1-Linux-x86_64.sh"
bash "Anaconda3-4.3.1-Linux-x86_64.sh" -b

echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all

# install cudnn libraries
wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"
tar -zxf cudnn.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

#install nvidia-docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb


#continue from here...install python 2.7 env
conda create -y -n py27 python=2.7 anaconda
source activate py27
conda upgrade -y --all

#install imagemagick for image transformations for data processing
sudo apt-get install imagemagick


# install and configure theano
pip install theano
echo "[global]
device = gpu
floatX = float32

[cuda]
root = /usr/local/cuda" > ~/.theanorc

#install keras, pytorch, editing .json for TF later
pip install keras==1.2.2
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
conda install -y pytorch torchvision cuda80 -c soumith
pip install keras-tqdm
source deactivate

#install python 3.6 environment for tensorflow, install tf, keras, update, pytorch
conda create -y -n k2 anaconda
conda upgrade -y --all
source activate k2
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp36-cp36m-linux_x86_64.whl
#keras
pip install keras
conda install -y pytorch torchvision cuda80 -c soumith
pip install keras-tqdm
mkdir ~/.keras
echo '{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}' > ~/.keras/keras.json
source deactivate



# configure jupyter and prompt for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
mkdir $HOME/.jupyter/
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
pip install keras-tqdm
sudo pip install keras-vis
jupyter contrib nbextension install --user
cd ~
git clone https://github.com/PAIR-code/facets
jupyter nbextension install facets-dist



#add conda envs
source activate py27

python -m ipykernel install --user --name py27 --display-name "Python2 (Keras1TheanoPT)"
pip install xgboost
conda install -y bcolz
easy_install --upgrade gensim
source deactivate

#for the other env
source activate k2

python -m ipykernel install --user --name k2 --display-name "Python3 (TensorflowK2.0PT)"
pip install xgboost
conda install -y bcolz
easy_install --upgrade gensim
source deactivate


# clone the fast.ai course repo and prompt to start notebook
cd ~
git clone https://github.com/fastai/courses.git
echo "\"jupyter notebook\" will start Jupyter on port 8888"
echo "If you get an error instead, try restarting your session so your $PATH is updated"
echo "Your environment with Python 2.7 and Theano, Keras1.2.2, and Pytorch is named: py27"
echo "Python 3.6 environment with TF, PyTorch, and Keras 2.0 is named: k2"
echo "Note: Keras is initially configured for TensorFlow, NOT Theano"