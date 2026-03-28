# Student_Grade_Predictor

A lightweight, interactive command-line tool that predicts a student's final grade based on their study habits and academic history. 

This project implements a K-Nearest Neighbors (KNN) machine learning algorithm entirely from scratch using only standard Python libraries. It generates synthetic training data, normalizes it to prevent scale bias, and uses Euclidean distance to find the most similar student profiles to predict your grade.

## Features

* **Zero External Dependencies:** Built purely with Python's standard math and random libraries.
* **Custom KNN Implementation:** Includes from-scratch functions for train/test splitting, Min-Max normalization and Euclidean distance calculation.
* **Interactive CLI:** Prompts the user to enter their own study metrics and instantly provides a predicted grade (A, B, C, D, or F).
* **Actionable Feedback:** Analyzes the user's inputs and provides customized study tips (e.g., advising more sleep or better attendance) to help improve their score.


## How to Run 

Since this project uses no external libraries, running it is incredibly simple.

**1. Clone or download the repository**
Ensure you have the `fundamental project.py` file on your local machine.

**2. Run the script**
Open your terminal, navigate to the folder containing the file and execute:


## How It Works
**1. Data Generation**
The script quietly generates 200 synthetic student profiles with varying study hours, attendance rates, previous scores, assignment completion and sleep hours.

**2. Training & Normalization**
It splits the data into training and testing sets and normalizes the features so that no single metric (like attendance out of 100) overpowers smaller metrics (like sleep out of 9).

**3. Evaluation**
It tests the model's accuracy on unseen data and prints the percentage before prompting the user.

**4. Prediction**
When you enter your details, the algorithm calculates the distance between your profile and all the profiles in the training data. It finds your 5 nearest neighbors (k=5) and predicts your grade based on a majority vote.


## Inputs Required
When prompted, you will need to provide the following details:

**1] Study hours per day** (1.0 to 10.0)

**2] Attendance percentage** (40.0 to 100.0)

**3] Previous exam score** (30.0 to 100.0)

**4] Assignments completed** (0 to 10)

**5] Sleep hours per night** (4.0 to 9.0)

