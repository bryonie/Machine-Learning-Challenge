#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

class Config:
    #def __init__(self, mode = 'conv', nfilt = 26, nfeat = 13, nfft = 4096, rate = 4000):
    def __init__(self, mode = 'conv', nfilt = 26, nfeat = 13, nfft = 512, rate = 16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')


# In[ ]:




