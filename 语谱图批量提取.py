#!/usr/bin/env python
# coding: utf-8

# In[2]:


#from pyAudioAnalysis import audioBasicIO
#from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
import os
import wave
import librosa
from librosa import display
import scipy
import h5py


# In[3]:


def extract_mel(path_to_audio):
    y, sr=librosa.load(path_to_audio, sr=None)
    y=scipy.signal.lfilter(([1,-0.97]),1,y)
    melspec=librosa.feature.melspectrogram(y,sr,n_fft=512, hop_length=256, n_mels=128)
    logmelspec= librosa.power_to_db(melspec)
    return logmelspec


# In[4]:


path = "../corpus/iemocap/"
emotions = os.listdir(path)#得到1级文件夹下的所有文件名称sad', 'xxx', 'hap'
print(emotions)


# In[ ]:


target_path = 'vary_128.hdf5'
dataset = h5py.File(target_path, 'w')


# In[6]:


for emotion in emotions:
    audios=os.listdir(os.path.join(path, emotion))
    c=0
    for audio in audios:
        c=c+1
        if c%100==0:
            print(str(c)+str(emotion)+' audios computed.')
        logmelspec= extract_mel(os.path.join(path, emotion, audio))
#         print(type(logmelspec), logmelspec.shape)
#         break
        target_key = audio[3:7]+emotion+'_'+audio[7:-4]+'.csv'
        dataset[target_key] = logmelspec


# In[ ]:


dataset.close()


# In[ ]:



