# Candidate Elimination Algorithm

import csv

# Read training data
data = []

with open("training_data.csv", "r") as file:
    reader = csv.reader(file)

    # Skip header
    next(reader)

    for row in reader:
        data.append(row)

# Number of attributes
num_attributes = len(data[0]) - 1

# Initialize Specific Hypothesis
S = ['0'] * num_attributes

# Initialize General Hypothesis
G = [['?'] * num_attributes]

print("Initial Specific Hypothesis:", S)
print("Initial General Hypothesis:", G)

# Process training examples
for row in data:

    attributes = row[:-1]
    target = row[-1]

    # Positive example
    if target == "Yes":

        for i in range(num_attributes):

            if S[i] == '0':
                S[i] = attributes[i]

            elif S[i] != attributes[i]:
                S[i] = '?'

    # Negative example
    elif target == "No":

        new_G = []

        for hypothesis in G:

            covers_negative = True

            for i in range(num_attributes):

                if hypothesis[i] != '?' and hypothesis[i] != attributes[i]:
                    covers_negative = False
                    break

            if not covers_negative:
                new_G.append(hypothesis)

        G = new_G

    # Display result for each example
    print("\nTraining Example:", row)
    print("Specific Hypothesis:", S)
    print("General Hypothesis:", G)

# Final result
print("\nFinal Version Space")
print("Specific Hypothesis:", S)
print("General Hypothesis:", G)