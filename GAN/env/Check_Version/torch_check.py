#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

print("python : ", str(sys.version_info[0]) +"."+ str(sys.version_info[1]) +"."+ str(sys.version_info[2]))


# In[2]:


import torch

print("PyTorch version: ", torch.__version__)
print("CUDA Version: ", torch.version.cuda)
print("cuDNN version: ", torch.backends.cudnn.version())


# In[ ]:




