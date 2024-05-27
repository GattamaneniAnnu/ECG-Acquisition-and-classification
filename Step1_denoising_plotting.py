import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft

# Load ECG signal data from CSV file
file_path = "farha_20_data1.csv"  # Replace with your CSV file path
ecg_data = pd.read_csv(file_path)
ecg_signal = ecg_data['ECG'].values

# Define wavelet transform parameters for denoising
wavelet = 'db4'  # Choose wavelet type
level = 5  # Level of decomposition

# Perform wavelet transform for denoising
coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

# Adaptive thresholding function for denoising
def soft_threshold(coefficients, threshold):
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0)

# Set adaptive threshold level based on median absolute deviation (MAD) for denoising
threshold = 0.6745 * np.median(np.abs(coeffs[-level]))

# Apply soft thresholding to detail coefficients for denoising
denoised_coeffs = [coeffs[0]] + [soft_threshold(coeff, threshold) for coeff in coeffs[1:]]

# Reconstruct denoised signal
denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

# QRS complex detection using approximation coefficients
approximation_coeffs = coeffs[0]

# Find peaks in approximation coefficients (QRS peaks)
peaks, _ = find_peaks(denoised_signal, height=0)

# ECG data processing for R peak detection
samplingrate = 20  # Sample rate in Hz

# Remove lower frequencies using FFT similar to MATLAB
fresult = fft(ecg_signal)
fresult[0:round(len(fresult)*5/samplingrate)] = 0
fresult[len(fresult)-round(len(fresult)*5/samplingrate):] = 0
corrected = np.real(ifft(fresult))

# Function for window-based maximum filter (similar to MATLAB ecgdemowinmax)
def ecgdemowinmax(signal, window_size):
    filtered = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        filtered[i] = np.max(signal[start:end])
    return filtered

# Calculate window size for filtering
WinSize = int(np.floor(samplingrate * 571 / 1000))
if WinSize % 2 == 0:
    WinSize += 1
filtered1 = ecgdemowinmax(corrected, WinSize)

# Scale ecg and filter by threshold
peaks1 = filtered1 / (max(filtered1) / 7)
peaks1[peaks1 < 4] = 0
peaks1[peaks1 >= 4] = 1
positions = np.nonzero(peaks1)[0]
distance = positions[1] - positions[0]

# Optimize filter window size
QRdistance = int(np.floor(0.04 * samplingrate))
if QRdistance % 2 == 0:
    QRdistance += 1
WinSize = 2 * distance - QRdistance

# Filter - second pass (similar to MATLAB function ecgdemowinmax)
filtered2 = ecgdemowinmax(corrected, WinSize)
peaks2 = filtered2.copy()
peaks2[peaks2 < 4] = 0
peaks2[peaks2 >= 4] = 1

# Extract R peak detection results for the specified range (40000 to 40500)
range_start = 40000
range_end = 40500
range_indices = np.where((peaks >= range_start) & (peaks <= range_end))[0]
r_peaks_range = peaks[range_indices]

# Plotting the original signal, denoised signal, and efficient R peaks for the specified range
plt.figure(figsize=(12, 8))

# Plot original ECG signal
plt.subplot(3, 1, 1)
plt.plot(ecg_signal[range_start:range_end], color='b')
plt.title('Original ECG Signal (40000-40500)')

# Plot denoised ECG signal
plt.subplot(3, 1, 2)
plt.plot(denoised_signal[range_start:range_end], color='r')
plt.title('Denoised ECG Signal (40000-40500)')

# Plot efficient R peaks on denoised signal
plt.subplot(3, 1, 3)
plt.plot(denoised_signal[range_start:range_end], color='g', label='Denoised ECG Signal')
plt.plot(r_peaks_range - range_start, denoised_signal[r_peaks_range], "x", color='r', label='Efficient R Peaks')
plt.legend()
plt.title('Efficient R Peak Detection (40000-40500)')

plt.tight_layout()
plt.show()

# Additional processing for R peak detection
positions2 = np.nonzero(peaks2)[0]
distanceBetweenFirstAndLastPeaks = positions2[-1] - positions2[0]
averageDistanceBetweenPeaks = distanceBetweenFirstAndLastPeaks / len(positions2)
averageHeartRate = 60 * samplingrate / averageDistanceBetweenPeaks
print('Average Heart Rate =', averageHeartRate)

# # Save filtered signal data and R-peaks to CSV
# output_data = pd.DataFrame({'ECG': corrected, 'Filtered1': filtered1, 'Peaks1': peaks1, 'Filtered2': filtered2, 'Peaks2': peaks2})
# output_data.to_csv('new_filtered_ecg_data_bujji.csv', index=False)
