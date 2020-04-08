# CUDA_ROOT=/usr/local/cuda-9.0
# TF_ROOT=/home/user/.local/lib/python3.6/site-packages/tensorflow

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') # /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow/include

# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') 	 # /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow

# $CUDA_ROOT/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 --expt-relaxed-constexpr && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -L $CUDA_ROOT/lib64 -O2

# echo 'nndistance'



CUDA_ROOT=/usr/local/cuda-10.0
TF_ROOT=/home/user/.local/lib/python3.6/site-packages/tensorflow

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') # /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow/include

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') 	 # /home/username/anaconda3/envs/py36_gpu/lib/python3.6/site-packages/tensorflow

$CUDA_ROOT/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -L $CUDA_ROOT/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

echo 'nndistance'
