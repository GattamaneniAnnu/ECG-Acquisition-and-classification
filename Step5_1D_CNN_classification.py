# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load the data from the Excel file
# data = pd.read_excel('r_peaks_data_with_clusters1_new.xlsx')
# X = data.iloc[:, :-1].values  # Features are all columns except the last one
# y = data.iloc[:, -1].values   # Labels are in the last column

# # Splitting the dataset into training and testing sets with stratified sampling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Reshaping the input data to fit the model (assuming each sample has 60 features)
# X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Defining the 1D CNN model
# model = Sequential([
#     Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compiling the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Training the model
# model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

# # Evaluating the model
# loss, accuracy = model.evaluate(X_test_reshaped, y_test)
# print(f'Test accuracy: {accuracy}')

# # Making predictions
# y_pred_proba = model.predict(X_test_reshaped)
# y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# # Displaying the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')


# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load the data from the Excel file
# data = pd.read_excel('r_peaks_data_with_clusters1.xlsx')
# X = data.iloc[:, :-2].values  # Features are all columns except the last one
# y = data.iloc[:, -1].values   # Labels are in the last column

# # Splitting the dataset into training and testing sets with stratified sampling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Reshaping the input data to fit the model (assuming each sample has 60 features)
# X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Define a more complex CNN model
# model = Sequential([
#     Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     MaxPooling1D(pool_size=2),
#     Conv1D(128, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),
#     Conv1D(256, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),  # Adding dropout layer with dropout rate of 0.5
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model with a lower learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # Train the model with a smaller batch size and more epochs
# model.fit(X_train_reshaped, y_train, epochs=150, batch_size=16, validation_split=0.2)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test_reshaped, y_test)
# print(f'Test accuracy: {accuracy}')

# # Making predictions
# y_pred_proba = model.predict(X_test_reshaped)
# y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# # Displaying the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# model.save('sample_model.h5')

# import pandas as pd

# # Load the data from the Excel file
# data = pd.read_excel('r_peaks_data_with_clusters1_new.xlsx')

# # Print the first few rows of the data
# print(data.head())

# # Print information about the data (columns, data types, etc.)
# print(data.info())

# # Describe the data (summary statistics)
# print(data.describe())

# # Check for any missing values in the data
# print(data.isnull().sum())


import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the Excel file
data = pd.read_excel('r_peaks_data_with_clusters1_new.xlsx')
X = data.iloc[:, :-1].values  # Features are all columns except the last one
y = data.iloc[:, -1].values   # Labels are in the last column

# Splitting the dataset into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Reshaping the input data to fit the model (assuming each sample has 60 features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Defining the 1D CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluating the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test accuracy: {accuracy}')

# Making predictions
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Displaying the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

model.save('sample_model.h5')