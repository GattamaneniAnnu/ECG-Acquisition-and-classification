import numpy as np
import pywt
import pandas as pd
from scipy.signal import find_peaks

# Load ECG signal data from CSV file
file_path = "farha_20_data1.csv"  # Replace with your CSV file path
ecg_data = pd.read_csv(file_path)
ecg_signal = ecg_data['ECG'].values
ecg_signal = ecg_signal[30000:70000]

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
peaks, _ = find_peaks(denoised_signal, height=650)  # Threshold set at 650 for R peaks detection

# Define window sizes for storing data
before_peak_window = 25
after_peak_window = 35

# Initialize list to store data rows
data_rows = []

# Iterate through each detected R peak
for peak_index in peaks:
    # Extract data around the R peak
    start_index = max(peak_index - before_peak_window, 0)
    end_index = min(peak_index + after_peak_window + 1, len(denoised_signal))
    data_row = denoised_signal[start_index:end_index]
    r_peak_value = denoised_signal[peak_index]  # Get the R peak value
    
    # Append the R peak value to the data row
    data_row = np.append(data_row, r_peak_value)
    
    # Append the modified data row to the list
    data_rows.append(data_row)

# Convert the list of data rows to a DataFrame
output_data = pd.DataFrame(data_rows)

# Rename the last column to 'rpeaks'
output_data = output_data.rename(columns={output_data.columns[-1]: 'rpeaks'})

# Save the data to Excel
output_data.to_excel('r_peaks_data_output.xlsx', index=False)
