import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

num_samples = 50000
fsamp = 1024

ecg_noisy = np.loadtxt('enel420_grp_11.txt')

#noise is ~32.6 Hz and ~61.7 Hz
fl_1 = 28.6
fh_1 = 36.6
fl_2 = 57.7
fh_2 = 65.7

#no need to normalise as fs is passed to firwin
notch_cutoff = [fl_1, fh_1, fl_2, fh_2]

#Filter frequency response
filter_coefficients = scipy.signal.firwin(numtaps=399, cutoff=notch_cutoff, window='hamming', fs=fsamp)
w, h = scipy.signal.freqz(filter_coefficients)
f = (fsamp*w)/(2*np.pi)
plt.figure(1)
plt.plot(f, 20*np.log10(abs(h)))
plt.xlim(0, 100)
plt.title('Filter Frequency Response')
plt.ylabel('Amplitude', color='b')
plt.show()

#Filter phase response
plt.figure(2)
plt.plot(f, np.angle(h))
plt.xlim(0, 40)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.title('Filter Phase Response')
plt.show()

#Filtered ECG signal
ecg_filtered = scipy.signal.lfilter(filter_coefficients, [1.0], ecg_noisy)
time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)
plt.figure(3)
plt.plot(time[0:len(time)-1], ecg_filtered)
plt.xlabel('Time (s)')
plt.ylabel('ECG Voltage (uV)')
plt.title('Filtered ECG Signal')
plt.show()

#FFT of filtered ECG
ecg_filt_fft = abs(scipy.fft.fft(ecg_filtered))
freq = scipy.fft.fftfreq(num_samples, 1/fsamp)
plt.figure(4)
plt.plot(freq, ecg_filt_fft)
plt.xlim(-200, 200)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.title('Spectrum of Filtered ECG Signal')
plt.show()
