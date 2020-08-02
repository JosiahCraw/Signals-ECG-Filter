import scipy
import numpy as np
import matplotlib.pyplot as plt

num_samples = 50000
fs = 1024

ecg_noisy = np.loadtxt('enel420_grp_11.txt')

ecg_fft = abs(scipy.fft.fft(ecg_noisy))
freq = scipy.fft.fftfreq(num_samples, 1/fs)

plt.plot(freq, ecg_fft)
plt.xlim(-200, 200)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.title('Spectrum of Noisy ECG Signal')
plt.show()
