import tkinter as tk
import numpy as np
import csv
import tensorflow as tf
from sudoku_solver import solve_sudoku

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


def prepare_input(puzzle_str):
    return string_to_array(puzzle_str).reshape(1, 81) / 9.0


def solve_puzzle_with_model(idx):
    this_quiz = quizzes[idx]
    prepared_input = prepare_input(this_quiz)
    predicted_solution = model.predict(prepared_input)
    solution = np.argmax(predicted_solution, axis=2).reshape(9, 9) + 1
    return solution


def solve_puzzle_traditional(idx):
    this_quiz = quizzes[idx]
    quiz_array = np.array(list(map(int, list(this_quiz)))).reshape(9, 9)
    grid = quiz_array.tolist()
    solve_sudoku(grid)
    return np.array(grid)


def display_puzzle(idx):
    root = tk.Tk()
    root.title(f"Sudoku Puzzle {idx + 1}")

    label_quiz = tk.Label(root, text="Quiz", font=("Arial", 12, "bold"))
    label_quiz.grid(row=0, column=0, columnspan=9)

    label_solution_model = tk.Label(root, text="Solution (Model)", font=("Arial", 12, "bold"))
    label_solution_model.grid(row=0, column=10, columnspan=9)

    label_solution_traditional = tk.Label(root, text="Solution (Traditional)", font=("Arial", 12, "bold"))
    label_solution_traditional.grid(row=0, column=20, columnspan=9)

    separator1 = tk.Frame(root, bg='red', width=5)
    separator1.grid(row=0, column=9, rowspan=10, padx=5)

    separator2 = tk.Frame(root, bg='blue', width=5)
    separator2.grid(row=0, column=19, rowspan=10, padx=5)

    this_quiz = quizzes[idx]
    quiz_array = np.array(list(map(int, list(this_quiz)))).reshape(9, 9)

    solution_model = solve_puzzle_with_model(idx)
    solution_traditional = solve_puzzle_traditional(idx)

    for n in range(9):
        for j in range(9):
            entry_quiz = tk.Entry(root, width=4, justify='center')
            entry_quiz.grid(row=n + 1, column=j, padx=2, pady=2)
            number = quiz_array[n][j]
            entry_quiz.insert(tk.END, 'x' if number == 0 else str(number))  # Replace 0's with 'x'
            if number != 0 and number == solution_traditional[n][j]:
                entry_quiz.config(state='readonly', foreground='red')  # Set matched numbers to red
            else:
                entry_quiz.config(state='readonly')

            entry_solution_model = tk.Entry(root, width=4, justify='center')
            entry_solution_model.grid(row=n + 1, column=j + 10, padx=2, pady=2)
            number_model = solution_model[n][j]
            entry_solution_model.insert(tk.END, str(number_model))
            if number_model != 0 and number_model == quiz_array[n][j]:
                entry_solution_model.config(state='readonly', foreground='red')  # Set matched numbers to red

            entry_solution_traditional = tk.Entry(root, width=4, justify='center')
            entry_solution_traditional.grid(row=n + 1, column=j + 20, padx=2, pady=2)
            number_traditional = solution_traditional[n][j]
            entry_solution_traditional.insert(tk.END, str(number_traditional))
            if number_traditional != 0 and number_traditional == quiz_array[n][j]:
                entry_solution_traditional.config(state='readonly', foreground='red')  # Set matched numbers to red

    root.mainloop()


# Display puzzles interactively
for i in range(len(quizzes)):
    display_puzzle(i)
