all:
	@echo "Usage: make [tex] [img] [show]"

tex:
	python fir_freq_sampled.py tex
	python fir_optimal.py tex
	python fir_window_lin_phase.py tex
	python iir_notch.py tex
	python noisy_ecg_plot_and_spectrum.py tex

img:
	python fir_freq_sampled.py img
	python fir_optimal.py img
	python fir_window_lin_phase.py img
	python iir_notch.py img
	python noisy_ecg_plot_and_spectrum.py img

show:
	python fir_freq_sampled.py
	python fir_optimal.py
	python fir_window_lin_phase.py
	python iir_notch.py
	python noisy_ecg_plot_and_spectrum.py