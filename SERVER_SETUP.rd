=====
to install on azure t4 server with ubuntu 22
make sure secure boot is disabled
make sure all ndes are in the same vnet and can connect to one another without password
=====

apt update
apt upgrade

pre install for nvidia driver+cuda
apt install -y build-essential
apt-get -y install pkg-config
apt-get -y install libglvnd-dev
apt install -y libvulkan1 mesa-vulkan-drivers

run https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup#install-cuda-drivers-on-n-series-vms
apt update && sudo apt install -y ubuntu-drivers-common
ubuntu-drivers install

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
apt install -y ./cuda-keyring_1.1-1_all.deb
apt update
apt -y install cuda-toolkit-12-2

restart
verify with
nvidia-smi
nvcc --version

to bashrc file
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64


run https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
sudo apt install libnccl2=2.18.5-1+cuda12.2 libnccl-dev=2.18.5-1+cuda12.2

test nccl:
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
make MPI=1
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

install miniconda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh
export PATH=/home/volumez/miniconda3/bin:$PATH
mkdir /data/miniconda3/pkgs /data/miniconda3/envs
conda config --add pkgs_dirs /data/miniconda3/pkgs
conda config --add envs_dirs /data/miniconda3/envs
conda install -c conda-forge mamba
