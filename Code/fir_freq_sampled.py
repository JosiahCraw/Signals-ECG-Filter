# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:29:43 2020

@author: jhu86
"""

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

num_samples = 50000
fsamp = 1024

ecg_noisy = np.loadtxt('enel420_grp_11.txt')

f_resp_ideal = np.ones(400)
f_resp_ideal[12:13+1] = 0
f_resp_ideal[24:25+1] = 0

impulse_resp = np.fft.ifft(f_resp_ideal)
h = np.real(impulse_resp)

h_shift = np.ones(400)
h_shift[0:200]=h[200:400]
h_shift[200:400]=h[0:200]

plt.plot(abs(np.fft.fft(h_shift)))
