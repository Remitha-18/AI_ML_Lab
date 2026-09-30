# Experiment 9
# Implementation of Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Training data
# [Study Hours, Attendance]
X = [
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
    [7, 90],
    [8, 95]
]

# Target
# 0 = Fail
# 1 = Pass
y = [0, 0, 0, 1, 1, 1, 1, 1]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display results
print("----- NAIVE BAYES CLASSIFICATION -----")

print("\nTest Data:")
print(X_test)

print("\nActual Output:")
print(y_test)

print("\nPredicted Output:")
print(y_pred.tolist())

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Predict a new student
new_student = [[6, 85]]

prediction = model.predict(new_student)

print("\nNew Student:")
print("Study Hours = 6")
print("Attendance = 85")

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")

print("\nNaive Bayes Classification Completed Successfully!")