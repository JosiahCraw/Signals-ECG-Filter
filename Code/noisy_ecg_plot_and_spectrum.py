import scipy
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_samples = 50000
    fs = 1024

    ecg_noisy = np.loadtxt('enel420_grp_11.txt')
    time = np.linspace(start=0, stop=num_samples/fs, num=num_samples+1)

    ecg_fft = abs(scipy.fft.fft(ecg_noisy))
    freq = scipy.fft.fftfreq(num_samples, 1/fs)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(time[0:len(time)-1], ecg_noisy)
    axs[0].set_title("Noisy ECG Signal")
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('ECG Voltage (uV)')

    axs[1].plot(freq, ecg_fft)
    axs[1].set_title("Noisy ECG Spectrum")
    axs[1].set_xlim(-100, 100)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Signal Power')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
