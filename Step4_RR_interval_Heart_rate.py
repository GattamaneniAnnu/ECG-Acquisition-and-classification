import pandas as pd
import numpy as np

# Load your data (replace this with your actual data loading)
data = pd.read_excel("r_peaks_data_output.xlsx")  # Assuming your data is in an Excel file

# Create a new column for RR-intervals
data['RR-intervals'] = 0.0

# Iterate over rows to compute RR-intervals
for i in range(1, len(data)):
    try:
        rr_interval = abs(data.at[i, 'rpeaks'] - data.at[i - 1, 'rpeaks'])
        data.at[i, 'RR-intervals'] = rr_interval
    except KeyError:
        print("KeyError: Make sure 'rpeaks' column exists in your DataFrame.")

# Calculate heart rate and add it as a new column
data['Heart Rate (bpm)'] = 6000 / data['RR-intervals']

# Print the updated DataFrame
print(data["RR-intervals"])
data.to_excel("output_data_2.xlsx", index=False)

data = pd.read_excel("output_data_2.xlsx")
average_heart_rate = data['Heart Rate (bpm)'].mean()
print(f"Average Heart Rate: {average_heart_rate} bpm")