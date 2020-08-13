import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import sys

num_samples = 50000
fsamp = 1024

def main():
    ecg_noisy = np.loadtxt('enel420_grp_11.txt')

    #Noise is ~32.6 Hz and ~61.7 Hz
    noise_band_1 = [32.5, 32.7]
    noise_band_2 = [61.6, 61.8]
    trans_width = 3.5

    #Specifying notch cutoff frequencies
    #No need to normalise as fs is passed to firwin
    notch_cutoff = [0, noise_band_1[0]-trans_width, noise_band_1[0], noise_band_1[1], noise_band_1[1]+trans_width,
                    noise_band_2[0]-trans_width, noise_band_2[0], noise_band_2[1], noise_band_2[1]+trans_width, fsamp/2]

    #Filter frequency response
    filter_coefficients = scipy.signal.remez(numtaps=399, bands=notch_cutoff, desired=[1, 0, 1, 0, 1], fs=fsamp)
    w, h = scipy.signal.freqz(filter_coefficients)
    f = (fsamp*w)/(2*np.pi)

    #Filtered ECG signal
    ecg_filtered = scipy.signal.lfilter(filter_coefficients, [1.0], ecg_noisy)
    time = np.linspace(start=0, stop=num_samples/fsamp, num=num_samples+1)

    #FFT of filtered ECG
    ecg_filt_fft = abs(scipy.fft.fft(ecg_filtered))
    freq = scipy.fft.fftfreq(num_samples, 1/fsamp)

    #Plotting filter frequency response, time-domain filtered ECG, and frequency-domain filtered ECG
    fig = plt.figure()
    axs = fig.subplots(3, 1)

    axs[0].plot(f, 20*np.log10(abs(h)))
    axs[0].set_xlim(0, 100)
    axs[0].set_title('Filter Frequency Response')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Amplitude (dB)')

    axs[1].plot(time[0:len(time)-1], ecg_filtered)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('ECG Voltage ($\mu V$)')
    axs[1].set_title('Filtered ECG Signal')

    axs[2].plot(freq, ecg_filt_fft)
    axs[2].set_xlim(-100, 100)
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('abs(Y(f)) ($\mu V^2$)')
    axs[2].set_title('Spectrum of Filtered ECG Signal')

    plt.tight_layout()
    if len(sys.argv) == 2:
        if sys.argv[1] == "tex":
            fig.savefig('../img/fir_optimal.pgf')
        if sys.argv[1] == "img":
            fig.savefig('../img/fir_optimal.png')
    else:
        fig.show()

    #Noise power estimates
    noise_power_total = np.var(ecg_noisy) - np.var(ecg_filtered)
    print("Total noise power = " + str(noise_power_total))

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
