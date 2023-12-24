# Sudoku Solver

This repository contains an AI-powered Sudoku Solver built using TensorFlow. It can solve Sudoku puzzles of varying difficulties using machine learning techniques.

## How to Use

### Training the Model

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/sudoku-solver-model.git
    cd sudoku-solver-model
    ```

2. Training Data:
    - Download the `train.csv` file .

3. Training Process:
    - Put the `train.csv` file in the root directory of the project.
    - Run the `train.py` file to train the model.
   ```bash
    py train.py
    ```

### Predicting Sudoku Puzzles

1. After training, the model will be saved as `sudoku_solver_model.h5`.
2. Use the `predict.py` file to solve Sudoku puzzles.
   ```bash
    py predict.py
    ```
## Sample Puzzles

Check the `quizzes.csv` file to input Sudoku puzzles you want to solve using the trained model.

## Resources

- [Google Drive - Train.csv](https://drive.google.com/file/d/1BqI83JA5aLBKgKUZ_zdxQ1cNkYWqwThf/view?usp=sharing)
- [Personal Bio](https://nassir.bio.link/)
- [Project Presentation Slides](https://docs.google.com/presentation/d/1ulsv_wwRMVfiNfuowVj6d0epHcPN21dYyypO8qZCAfI/edit?usp=sharing)
- [Sudoku Solver Model Repository](https://github.com/NacirChahine/sudoku-solver-mode)

## Contributors

Developed, prepared, and presented by [Nacir Chahine](https://nassir.bio.link/).

