import numpy as np
import pandas as pd
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('sudoku_solver_model.h5')


# Function to convert string quiz to numpy array
def string_to_array(s):
    return np.array(list(map(int, list(s))))


# Sample Sudoku puzzles
quizzes = [
    '004300209005009001070060043006002087190007400050083000600000105003508690042910300',
    '006000085104008000000403100900025300000030000002890004003501000000700906570000800',
]


# Convert puzzles to the required format for prediction
def prepare_input(puzzle):
    return string_to_array(puzzle).reshape(1, 81) / 9.0


# Predict solutions for each puzzle
for puzzle in quizzes:
    # Prepare input
    prepared_input = prepare_input(puzzle)

    # Make prediction
    predicted_solution = model.predict(prepared_input)

    # Convert the predicted solution to the expected format
    solution = np.argmax(predicted_solution, axis=2).reshape(9, 9) + 1

    # Print the solved Sudoku puzzle
    print(f"Solved Puzzle:\n{solution}\n")
