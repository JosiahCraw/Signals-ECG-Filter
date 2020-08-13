import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import sys

num_samples = 50000
fsamp = 1024
num_coefficients = 400
num_transition_samples = 2

def main():
    ecg_noisy = np.loadtxt('enel420_grp_11.txt')

    spacing = np.linspace(0, fsamp, num_coefficients)

    f_resp_ideal = np.ones(num_coefficients)
    f_resp_ideal[12:13+1] = 0
    f_resp_ideal[24:25+1] = 0
    f_resp_ideal[387:388+1] = 0
    f_resp_ideal[375:376+1] = 0

    i = 0
    while i < num_transition_samples:
        f_resp_ideal[12-num_transition_samples+i] = (num_transition_samples-i+1)/(num_transition_samples+1)
        f_resp_ideal[13+num_transition_samples-i] = (num_transition_samples-i+1)/(num_transition_samples+1)
        f_resp_ideal[24-num_transition_samples+i] = (num_transition_samples-i+1)/(num_transition_samples+1)
        f_resp_ideal[25+num_transition_samples-i] = (num_transition_samples-i+1)/(num_transition_samples+1)
        i += 1

    impulse_resp = scipy.fft.ifft(f_resp_ideal)
    h = np.real(impulse_resp)

    h_shift = np.ones(num_coefficients)
    h_shift[0:int(num_coefficients/2)]=h[int(num_coefficients/2):num_coefficients]
    h_shift[int(num_coefficients/2):num_coefficients]=h[0:int(num_coefficients/2)]
    h_wind=h_shift*np.hamming(num_coefficients)

    #Filter frequency response
    w, h = scipy.signal.freqz(h_wind)
    f = (fsamp*w)/(2*np.pi)

    #Filtered ECG signal
    ecg_filtered = scipy.signal.lfilter(h_wind, [1.0], ecg_noisy)
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
            plt.savefig('../img/fir_freq.pgf')
        elif sys.argv[1] == "img":
            plt.savefig('../img/fir_freq.png')
    else:
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
