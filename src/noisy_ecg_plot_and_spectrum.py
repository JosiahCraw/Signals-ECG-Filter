import scipy
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import sys

num_samples = 50000
fs = 1024

def main():
    ecg_noisy = np.loadtxt('enel420_grp_11.txt')

    #Generating time axis for ECG plot
    time = np.linspace(start=0, stop=num_samples/fs, num=num_samples+1)
    
    #FFT of noisy signal and generating frequency axis
    ecg_fft = abs(scipy.fft.fft(ecg_noisy))
    freq = scipy.fft.fftfreq(num_samples, 1/fs)

    #Plotting noisy ECG and its spectrum
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(time[0:len(time)-1], ecg_noisy)
    axs[0].set_title("Noisy ECG Signal")
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('ECG Voltage ($\mu V$)')

    axs[1].plot(freq, ecg_fft)
    axs[1].set_title("Noisy ECG Spectrum")
    axs[1].set_xlim(-100, 100)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('abs(X(f)) ($\mu V^2$)')

    plt.tight_layout()
    if len(sys.argv) == 2:
        if sys.argv[1] == "tex":
            plt.savefig('../img/noisy.pgf')
        elif sys.argv[1] == "img":
            plt.savefig('../img/noisy.png')
    else:
        plt.show()

    #Plotting noisy ECG PSD (to estimate relative noise powers)
    f, psd = scipy.signal.periodogram(ecg_noisy, 1024)
    plt.figure(3)
    plt.plot(f, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density of Noisy ECG')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "tex":
            plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [],
            "pgf.texsystem": "pdflatex",
            'pgf.rcfonts': False})

    main()
