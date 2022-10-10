#!/usr/bin/env python
# coding: utf-8

# #### 2019年12月18日15点04分更新  
# 1.能量而非幅值  
# 2.线性而非对数  
# ~~~python
# #vocal,excitation= vocal.real,excitation.real
# 
# vocal,excitation=np.exp(vocal),np.exp(excitation)
# vocal,excitation=np.abs(vocal)**2,np.abs(excitation)**2
# ~~~

# #### 2019年12月18日18点38分更新  
# 修正了转到mel谱后忘记取对数的bug
# #### 2020年11月11日21点21分  
# n mels换成128，存到一个hdf5  
# #### 2020年11月12日11点10分  
# 恢复数据类型为float32，与原始谱一致

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import os
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.fftpack import fft, ifft
import h5py


# In[ ]:


def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    # zeros+=1e-8
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵
    return frames


# ### 2019年12月24日14点40分  
# 加窗

# In[ ]:


def devideExcitationAndVocal(y=None):
    l = len(y)
    y = y * np.hanning(l)  #############加窗
    Y = np.log(np.abs(fft(y)) + 1e-8)
    z = ifft(Y)
    mcep = 29
    zy = z[:mcep + 1]
    zy = np.hstack([np.matrix(zy), np.zeros([1, l - 2 * mcep - 1]), np.matrix(zy[-1:0:-1])])
    ZY = fft(zy)

    ft = np.hstack([np.zeros([1, mcep + 1]), np.matrix(z[mcep + 2:-mcep + 1]).H.T, np.zeros([1, mcep])])
    FT = fft(ft)

    return ZY[:, :l // 2], FT[:, :l // 2]  # 单边fft即可


# In[ ]:


path = "../corpus/iemocap/"
emotions = os.listdir(path)  # 得到1级文件夹下的所有文件名称sad', 'xxx', 'hap'
print(emotions)

# In[ ]:


exc_path = 'excVary_128.hdf5'
exc_dataset = h5py.File(exc_path, 'w')
voc_path = 'vocVary_128.hdf5'
voc_dataset = h5py.File(voc_path, 'w')

# In[ ]:


import time

t1 = time.time()
c = 0
for emotion in emotions:
    audios = os.listdir(path + emotion + '/')

    for audio in audios:
        c = c + 1
        if c % 100 == 0:
            t2 = time.time()
            interval = round(t2 - t1, 2)
            print(str(c) + ' audios computed, ' + str(interval) + 'seconds used.')
        y, sr = librosa.load(path + emotion + '/' + audio, sr=None)
        y = scipy.signal.lfilter(([1, -0.97]), 1, y)
        ys = enframe(y, 512, 256)
        vocal, excitation = np.zeros([ys.shape[0], 256], dtype='complex'), np.zeros([ys.shape[0], 256], dtype='complex')
        for i, y in enumerate(ys):
            ZY, FT = devideExcitationAndVocal(y)
            vocal[i, :] = ZY
            excitation[i, :] = FT
        vocal, excitation = vocal.T, excitation.T
        # vocal,excitation= vocal.real,excitation.real

        vocal, excitation = np.exp(vocal), np.exp(excitation)
        vocal, excitation = np.abs(vocal) ** 2, np.abs(excitation) ** 2
        vocal = librosa.feature.melspectrogram(sr=sr, S=vocal, n_mels=128)
        excitation = librosa.feature.melspectrogram(sr=sr, S=excitation, n_mels=128)  # to mel scale
        vocal = librosa.power_to_db(vocal, ref=np.max)
        excitation = librosa.power_to_db(excitation, ref=np.max)

        key = audio[3:7] + emotion + '_' + audio[7:-4] + '.csv'
        exc_dataset[key] = excitation.astype('<f4')
        voc_dataset[key] = vocal.astype('<f4')
        # break
t2 = time.time()
interval = round(t2 - t1, 2)
print('all audios computed, ' + str(interval) + 'seconds used.')

# In[ ]:


# In[ ]:
