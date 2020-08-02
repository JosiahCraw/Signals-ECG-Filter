import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

num_samples = 50000
fsamp = 1024

ecg_noisy = np.loadtxt('enel420_grp_11.txt')

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
