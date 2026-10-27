# Lab Sheet-08 — Java Implementations
Projects from the supplied lab sheet:
1. Handwritten Digit Recognition using a small neural network (MNIST CSV input).
2. Q-Learning based Maze Solver.

## Requirements
- JDK 17 or newer
- For digit recognition, download the MNIST CSV files and place them in this folder:
  - `mnist_train.csv`
  - `mnist_test.csv`
  Each row must contain 785 comma-separated values: label (0–9), then 784 pixel values (0–255).
  This program uses a small feed-forward neural network trained with backpropagation; it is a teaching implementation and may take time on the full dataset.
- No third-party Java libraries are required.

## Compile and run (Git Bash / terminal)
```bash
javac MnistDigitRecognizer.java
java MnistDigitRecognizer mnist_train.csv mnist_test.csv
```
```bash
javac QLearningMazeSolver.java
java QLearningMazeSolver
```
The digit program saves `digit_model.txt`; the maze program prints the learned route and episode rewards.
