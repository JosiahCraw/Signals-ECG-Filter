import scipy.signal
import scipy
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

#Filtered ECG signal
ecg_notch_1 = scipy.signal.lfilter(B_n1, A_n1, ecg_noisy)
ecg_notch_2 = scipy.signal.lfilter(B_n2, A_n2, ecg_noisy)
ecg_final = scipy.signal.lfilter(B_n2, A_n2, ecg_notch_1)
time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)

#FFT of filtered ECG
ecg_final_fft = abs(scipy.fft.fft(ecg_final))
freq = scipy.fft.fftfreq(num_samples, 1/fsamp)

fig, axs = plt.subplots(3, 1)

axs[0].plot(f, 20*np.log10(abs(cascaded_notch)))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Magnitude (dB)')
axs[0].set_title('Magnitude Response of Cascaded IIR Notch Filters')
axs[0].set_xlim(0, 100)

axs[1].plot(time[0:len(time)-1], ecg_final)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('ECG Voltage (uV)')
axs[1].set_title('Final Filtered ECG Signal')

axs[2].plot(freq, ecg_final_fft)
axs[2].set_xlim(-100, 100)
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('abs(Y(f)) ($uV^2$)')
axs[2].set_title('Spectrum of Filtered ECG Signal')

plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(time[0:len(time)-1], ecg_final)
plt.plot(time[0:len(time)-1], ecg_noisy)
plt.xlabel('Time (s)')
plt.ylabel('ECG Voltage (uV)')
plt.title('Delay of Cascaded IIR Notch Filters')
plt.xlim(10,11)
plt.show()

noise_power_32_6hz = np.var(ecg_noisy) - np.var(ecg_notch_1)
noise_power_61_7hz = np.var(ecg_noisy) - np.var(ecg_notch_2)
print("61.7 Hz noise power = " + str(noise_power_61_7hz))
print("32.6 Hz noise power = " + str(noise_power_32_6hz))
