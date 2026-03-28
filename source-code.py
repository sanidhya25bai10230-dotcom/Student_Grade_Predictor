import random
import math

random.seed(42)

FEATURES = ["study_hours", "attendance", "prev_score", "assignments", "sleep_hours"]


def generate_data(n=200):
    data = []
    for _ in range(n):
        study_hours = round(random.uniform(1, 10), 1)
        attendance  = round(random.uniform(40, 100), 1)
        prev_score  = round(random.uniform(30, 100), 1)
        assignments = random.randint(0, 10)
        sleep_hours = round(random.uniform(4, 9), 1)

        score = (
            study_hours * 4.0 +
            attendance  * 0.3 +
            prev_score  * 0.3 +
            assignments * 2.0 +
            sleep_hours * 1.0 +
            random.uniform(-8, 8)
        )

        if score >= 75:
            grade = "A"
        elif score >= 60:
            grade = "B"
        elif score >= 45:
            grade = "C"
        elif score >= 30:
            grade = "D"
        else:
            grade = "F"

        data.append({
            "study_hours": study_hours,
            "attendance":  attendance,
            "prev_score":  prev_score,
            "assignments": assignments,
            "sleep_hours": sleep_hours,
            "grade":       grade
        })
    return data


def train_test_split(data, test_ratio=0.2):
    shuffled = data[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def get_min_max(train_data):
    mins = {f: min(r[f] for r in train_data) for f in FEATURES}
    maxs = {f: max(r[f] for r in train_data) for f in FEATURES}
    return mins, maxs


def normalize_dataset(data, mins, maxs):
    normalized = []
    for row in data:
        nr = {f: (row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in FEATURES}
        
        if "grade" in row:
            nr["grade"] = row["grade"]
        normalized.append(nr)
    return normalized


def euclidean(a, b):
    return math.sqrt(sum((a[f] - b[f]) ** 2 for f in FEATURES))


def knn_predict(train_data, new_point, k=5):
    distances = []
    for row in train_data:
        dist = euclidean(row, new_point)
        distances.append((dist, row["grade"]))
    
    distances.sort(key=lambda x: x[0])


    top_k = [g for _, g in distances[:k]]
    
  
    votes = {}
    for g in top_k:
        votes[g] = votes.get(g, 0) + 1

    return max(votes, key=votes.get), votes


def evaluate(train_data, test_data, k=5):
    correct = 0
    for row in test_data:
        pred, _ = knn_predict(train_data, row, k)
        if pred == row["grade"]:
            correct += 1
    return round(correct / len(test_data) * 100, 2)


def get_tips(study_hours, attendance, assignments, sleep_hours):
    tips = []
    if study_hours < 3:
        tips.append("Try to study at least 4 to 5 hours per day.")
    if attendance < 75:
        tips.append("Your attendance is low. Try not to miss classes.")
    if assignments < 6:
        tips.append("Complete more assignments. They help build understanding.")
    if sleep_hours < 6:
        tips.append("Try to get 7 to 8 hours of sleep. It helps with memory and focus.")
    if not tips:
        tips.append("Your habits look good. Keep it up.")
    return tips


def predict_for_student(train_data, mins, maxs):
    print("\nEnter your details to get a grade prediction.")

    try:
        study_hours = float(input("Study hours per day (1 to 10): "))
        attendance  = float(input("Attendance percentage (40 to 100): "))
        prev_score  = float(input("Previous exam score (30 to 100): "))
        assignments = float(input("Assignments completed out of 10: "))
        sleep_hours = float(input("Sleep hours per night (4 to 9): "))
    except ValueError:
        print("Please enter numbers only.")
        return

    student = {
        "study_hours": study_hours,
        "attendance":  attendance,
        "prev_score":  prev_score,
        "assignments": assignments,
        "sleep_hours": sleep_hours
    }

    norm_student = normalize_dataset([student], mins, maxs)[0]
    
   
    grade, votes = knn_predict(train_data, norm_student, k=5)

    grade_labels = {
        "A": "Excellent",
        "B": "Good",
        "C": "Average",
        "D": "Below Average",
        "F": "Fail"
    }

    print("\nResult")
    print("Predicted Grade :", grade, "-", grade_labels.get(grade, ""))
    print("Votes from nearest neighbours :", votes)

    tips = get_tips(study_hours, attendance, assignments, sleep_hours)
    print("\nSuggestions for you:")
    for tip in tips:
        print(" -", tip)


def main():
    print("Welcome to the Student Grade Predictor")

    data = generate_data(n=200)
    train_raw, test_raw = train_test_split(data, test_ratio=0.2)
    mins, maxs = get_min_max(train_raw)
    
    train_data = normalize_dataset(train_raw, mins, maxs)
    
    
    while True:
        predict_for_student(train_data, mins, maxs)
        again = input("\nPredict for another student? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nThank you for using the Student Grade Predictor.")


if __name__ == "__main__":
    main()
