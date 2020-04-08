echo -e '\n\n========== system environment ============'

echo -e '\nCPU'
cat /proc/cpuinfo | grep "model name"


echo -e '\nGPU'
echo '0221' | sudo lshw -C video | awk -F'product: ' '/product/{print $2}'

echo -e '\n'
python --version

echo -e '\nnvidia'
nvidia-smi | grep "Driver Version" | gawk '{print $6}' | cut -c1-

echo -e '\ncuda toolkit'
nvcc --version | grep "release" | awk '{print $6}'

echo -e '\ncudnn'
locate cudnn | grep "libcudnn.so." | tail -n1 | sed -r 's/^.*\.so\.//'

echo -e '\ntensorflow'
python -c 'import tensorflow as tf; print(tf.__version__)'

echo -e '\npytorch'
python -c 'import torch as tc; print(tc.__version__)'
echo -e '\n'
