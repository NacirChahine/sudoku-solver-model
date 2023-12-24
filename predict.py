import numpy as np
import csv
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('sudoku_solver_model.h5')


# Function to convert string quiz to numpy array
def string_to_array(s):
    return np.array(list(map(int, list(s))))


# Initialize an empty list to store the quizzes
quizzes = []

# Path to the CSV file containing the quizzes
csv_file_path = 'quizzes.csv'  # Replace this with the actual path to your CSV file

# Read the quizzes from the CSV file
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Skip the header row if it exists
    quiz_column_index = header.index('quizzes')  # Find the index of 'quizzes' column

    for row in csv_reader:
        quiz = row[quiz_column_index]
        quizzes.append(quiz)

# Display the quizzes read from the CSV file
# for idx, quiz in enumerate(quizzes, start=1):
#     print(f"Quiz {idx}: {quiz}")


# Convert puzzles to the required format for prediction
def prepare_input(puzzle_str):
    return string_to_array(puzzle_str).reshape(1, 81) / 9.0


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
