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

#Filtered ECG signal
ecg_filtered = scipy.signal.lfilter(filter_coefficients, [1.0], ecg_noisy)
time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)

#FFT of filtered ECG
ecg_filt_fft = abs(scipy.fft.fft(ecg_filtered))
freq = scipy.fft.fftfreq(num_samples, 1/fsamp)

fig, axs = plt.subplots(3, 1)

axs[0].plot(f, 20*np.log10(abs(h)))
axs[0].set_xlim(0, 100)
axs[0].set_title('Filter Frequency Response')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Amplitude (dB)')

axs[1].plot(time[0:len(time)-1], ecg_filtered)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('ECG Voltage (uV)')
axs[1].set_title('Filtered ECG Signal')

axs[2].plot(freq, ecg_filt_fft)
axs[2].set_xlim(-100, 100)
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('abs(Y(f)) ($uV^2$)')
axs[2].set_title('Spectrum of Filtered ECG Signal')

plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(time[0:len(time)-1], ecg_filtered)
plt.plot(time[0:len(time)-1], ecg_noisy)
plt.xlabel('Time (s)')
plt.ylabel('ECG Voltage (uV)')
plt.title('Delay of Window FIR Filter')
plt.xlim(10,11)
plt.show()

noise_power = np.var(ecg_noisy) - np.var(ecg_filtered)
print(noise_power)




