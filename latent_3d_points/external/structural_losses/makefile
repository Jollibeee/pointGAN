
#/bin/bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudalib=/usr/local/cuda-10.0/lib64
cudainc=/usr/local/cuda-10.0/include
nsync=/home/sohee/anaconda3/envs/python3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=/home/sohee/anaconda3/envs/python3/lib/python3.6/site-packages/tensorflow/include
TF_LIB=/home/sohee/anaconda3/envs/python3/lib/python3.6/site-packages/tensorflow



## https://github.com/optas/latent_3d_points/issues/21

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(TF_INC) -I $(cudainc) -I $(nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(TF_INC) -I $(cudainc) -I $(nsync) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 --expt-relaxed-constexpr
 
tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(TF_INC) -I $(cudainc) -I $(nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(TF_INC) -I $(cudainc) -I $(nsync) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 --expt-relaxed-constexpr

clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 


# all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o

# tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
# 	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(TF_INC) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
# 	$(nvcc) -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
# 	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(TF_INC) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# tf_nndistance_g.cu.o: tf_nndistance_g.cu
# 	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# clean:
# 	rm tf_approxmatch_so.so
# 	rm tf_nndistance_so.so
# 	rm  *.cu.o 