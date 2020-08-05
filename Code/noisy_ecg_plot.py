import scipy
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_samples = 50000
    fs = 1024

    ecg_noisy = np.loadtxt('enel420_grp_11.txt')
    time = np.linspace(start=0, stop=num_samples/fs, num=num_samples+1)

    plt.plot(time[0:len(time)-1], ecg_noisy)
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Voltage (uV)')
    plt.title('Noisy ECG Signal')
    plt.show()

if __name__ == "__main__":
    main()
