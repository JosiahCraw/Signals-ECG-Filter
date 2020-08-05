import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

num_samples = 50000
fsamp = 1024

ecg_noisy = np.loadtxt('enel420_grp_11.txt')

#Coefficients for 32.6 Hz notch
A_n1 = np.array([1,-1.93005,0.969556])
B_n1 = np.array([0.99057,-1.94164,0.99057])

#Coefficients for 61.7 Hz notch
A_n2 = np.array([1,-1.82987,0.969556])
B_n2 = np.array([0.98633,-1.83297,0.98633])

#Compute frequency responses of the notch filters
w, h_1 = scipy.signal.freqz(B_n1, A_n1)
w, h_2 = scipy.signal.freqz(B_n2, A_n2)
f = fsamp*w/(2*np.pi)

#Cascade filters by multiplying in frequency
cascaded_notch = h_1*h_2

#Plot magnitude response
plt.figure(1)
plt.plot(f, 20*np.log10(abs(cascaded_notch)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Magnitude Response of Cascaded IIR Notch Filters')
plt.xlim(0, 100)
plt.show()

#Filtered ECG signal
ecg_notch_1 = scipy.signal.lfilter(B_n1, A_n1, ecg_noisy)
ecg_final = scipy.signal.lfilter(B_n2, A_n2, ecg_notch_1)
time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)
plt.figure(3)
plt.plot(time[0:len(time)-1], ecg_final)
plt.xlabel('Time (s)')
plt.ylabel('ECG Voltage (uV)')
plt.title('Final Filtered ECG Signal')
plt.show()

#FFT of filtered ECG
ecg_final_fft = abs(scipy.fft.fft(ecg_final))
freq = scipy.fft.fftfreq(num_samples, 1/fsamp)
plt.figure(4)
plt.plot(freq, ecg_final_fft)
plt.xlim(-200, 200)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.title('Spectrum of Filtered ECG Signal')
plt.show()
