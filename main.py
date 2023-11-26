import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('APTEC_PC10_CardBoard_2023_v1.csv')

# Select the CMYK columns as training data
training_data = df[['CMYK_C', 'CMYK_M', 'CMYK_Y']].values

# Select the LAB columns as target data
target_data = df[['LAB_L', 'LAB_A', 'LAB_B']].values

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
training_data_normalized = scaler.fit_transform(training_data)
target_data_normalized = scaler.fit_transform(target_data)

# Define the neural network structure
model = Sequential()
model.add(Dense(6, input_dim=4, activation='sigmoid'))  # 4 input dimensions for CMYK, 6 units in the hidden layer
model.add(Dense(3, activation='sigmoid'))  # 3 output dimensions for LAB

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(training_data_normalized, target_data_normalized, epochs=1000, batch_size=10, verbose=1)

# Plotting the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Printing final evaluation
final_loss = history.history['loss'][-1]
print(f'Final training loss: {final_loss}')

# Predicting from the training data (for visualization purpose)
predicted_target_data = model.predict(training_data_normalized)

# Visualizing some predictions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(target_data_normalized[:, 0], target_data_normalized[:, 1], c='red', label='Actual')
plt.scatter(predicted_target_data[:, 0], predicted_target_data[:, 1], c='blue', label='Predicted')
plt.title('Actual vs Predicted L*a*')
plt.xlabel('Normalized L*')
plt.ylabel('Normalized a*')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(target_data_normalized[:, 1], target_data_normalized[:, 2], c='red', label='Actual')
plt.scatter(predicted_target_data[:, 1], predicted_target_data[:, 2], c='blue', label='Predicted')
plt.title('Actual vs Predicted a*b*')
plt.xlabel('Normalized a*')
plt.ylabel('Normalized b*')
plt.legend()

plt.show()
