import pandas as pd
import numpy as np
import tensorflow as tf

# Load the data from the Excel file
data = pd.read_excel('r_peaks_data_with_clusters1.xlsx')

# Assuming 'input_row_index' is the index of the row you want to predict
input_row_index = 0  # Change this to the desired row index
input_row = data.iloc[input_row_index, :-2].values  # Exclude the last two columns

# Preprocess the input data to match the model's input shape
input_row_reshaped = input_row.reshape(1, input_row.shape[0], 1)  # Reshape for Conv1D input shape

# Load the trained model from the HDF5 file
model = tf.keras.models.load_model('sample_model.h5')  # Replace 'your_trained_model.h5' with the actual model file

# Predict the class (0 or 1)
y_pred_proba = model.predict(input_row_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Display the predicted class
print(f'Predicted Class: {y_pred[0]}')
