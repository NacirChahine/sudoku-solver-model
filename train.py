import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the data
data = pd.read_csv('train.csv')

# Extract quizzes and solutions
quizzes = data['quizzes'].values
solutions = data['solutions'].values

# Convert strings to numpy arrays
def string_to_array(s):
    return np.array(list(map(int, list(s))))

X = np.array([string_to_array(q) for q in quizzes])
y = np.array([string_to_array(s) for s in solutions])

# Normalize data
X = X / 9.0
y = y - 1  # Adjust to start from 0 (for model prediction)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape y_train and y_val to match the model's output shape for each cell
y_train_reshaped = np.array([tf.keras.utils.to_categorical(y, num_classes=9) for y in y_train])
y_val_reshaped = np.array([tf.keras.utils.to_categorical(y, num_classes=9) for y in y_val])

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((9, 9, 1), input_shape=(81,)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(81 * 9, activation='softmax'),  # Output size adjusted for 81 cells with 9 classes each
    tf.keras.layers.Reshape((81, 9))  # Reshape to match the target shape
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_reshaped, validation_data=(X_val, y_val_reshaped), epochs=10, batch_size=32)

# Define the file name for the model
model_file_name = 'sudoku_solver_model.h5'

# Save the model
model.save(model_file_name)

# # Load the saved model for making predictions
# loaded_model = tf.keras.models.load_model(model_file_name)
#
# # Example: Predict solution for a new Sudoku puzzle
# new_puzzle = np.array([0, 0, 4, 3, 0, 0, 2, 0, 9, ...])  # Replace with your Sudoku puzzle
# normalized_puzzle = new_puzzle / 9.0
# reshaped_puzzle = normalized_puzzle.reshape((1, 81))
# predicted_solution = loaded_model.predict(reshaped_puzzle)
#
# # Convert the predicted solution back to the original format if needed
# predicted_solution = np.argmax(predicted_solution, axis=1) + 1
# predicted_solution = predicted_solution.reshape((9, 9))
#
# print("Predicted solution:")
# print(predicted_solution)
