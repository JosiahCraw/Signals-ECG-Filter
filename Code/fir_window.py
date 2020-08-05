import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_samples = 50000
    fsamp = 1024

    ecg_noisy = np.loadtxt('enel420_grp_11.txt')

<<<<<<< HEAD
#noise is ~32.6 Hz and ~61.7 Hz
fl_1 = 28.6
fn_1 = 32.6
fh_1 = 36.6
fl_2 = 57.7
fn_2 = 61.7
fh_2 = 65.7

notch_cutoff = [0, fl_1, fn_1, fh_1, fl_2, fn_2, fh_2, fsamp/2]
A = [1, 1, 0, 1, 1, 0, 1, 1]

filter_coefficients = scipy.signal.firwin2(numtaps=400, freq=notch_cutoff, gain=A, fs=fsamp)
w, h = scipy.signal.freqz(filter_coefficients)
f = (fsamp*w)/(2*np.pi)

plt.plot(f, 20 * np.log10(abs(h)), 'b')
plt.xlim(0, 100)
plt.title('Notch Filter Frequency Response')
plt.ylabel('Amplitude (dB)', color='b')
plt.xlabel('Frequency (Hz)')
plt.show()

ecg_filtered = scipy.signal.lfilter(filter_coefficients, [1.0], ecg_noisy)
time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)

plt.plot(time[0:len(time)-1], ecg_filtered)
plt.xlabel('Time (s)')
plt.ylabel('ECG Voltage (uV)')
plt.title('Filtered ECG Signal')
plt.show()

ecg_filt_fft = abs(scipy.fft.fft(ecg_filtered))
freq = scipy.fft.fftfreq(num_samples, 1/fsamp)

plt.plot(freq, ecg_filt_fft)
plt.xlim(-200, 200)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.title('Spectrum of Filtered ECG Signal')
plt.show()
=======
    #noise is ~32.55-32.75 Hz and ~61.7 Hz
    fl = 32.55
    fh = 32.75

    #normalising cutoff frequencies to fs/2
    fl_norm = fl/(fsamp/2)
    fh_norm = fh/(fsamp/2)

    notch_cutoff = [fl_norm, fh_norm]

    filter_coefficients = scipy.signal.firwin(numtaps=399, cutoff=notch_cutoff, window='hamming', fs=fsamp)
    n = range(-199, 200, 1)

    print(filter_coefficients[198])
    print(filter_coefficients[199])
    print(filter_coefficients[200])

    plt.plot(n, filter_coefficients)
    plt.show()
    

if __name__ == "__main__":
    main()
>>>>>>> 4750fffde82244508ffc5a1046ccd45c2b7f432a
