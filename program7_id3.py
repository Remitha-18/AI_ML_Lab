# Decision Tree Learning using ID3 Algorithm

import pandas as pd
import math


# Calculate Entropy
def entropy(data):

    total = len(data)

    if total == 0:
        return 0

    counts = data["Play"].value_counts()

    entropy_value = 0

    for count in counts:
        probability = count / total
        entropy_value -= probability * math.log2(probability)

    return entropy_value


# Calculate Information Gain
def information_gain(data, attribute):

    total_entropy = entropy(data)

    values = data[attribute].unique()

    weighted_entropy = 0

    for value in values:

        subset = data[data[attribute] == value]

        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy


# ID3 Algorithm
def id3(data, attributes):

    # If all examples have same class
    if len(data["Play"].unique()) == 1:
        return data["Play"].iloc[0]

    # If no attributes are left
    if len(attributes) == 0:
        return data["Play"].mode()[0]

    # Find attribute with highest information gain
    gains = {}

    for attribute in attributes:
        gains[attribute] = information_gain(data, attribute)

    best_attribute = max(gains, key=gains.get)

    tree = {best_attribute: {}}

    remaining_attributes = [
        attribute for attribute in attributes
        if attribute != best_attribute
    ]

    # Create branches
    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        tree[best_attribute][value] = id3(
            subset,
            remaining_attributes
        )

    return tree


# Read dataset
data = pd.read_csv("id3_data.csv")

print("Training Data:")
print(data)

# Attributes
attributes = list(data.columns[:-1])

# Build Decision Tree
decision_tree = id3(data, attributes)

print("\nDecision Tree:")
print(decision_tree)

# Display Information Gain
print("\nInformation Gain:")

for attribute in attributes:
    gain = information_gain(data, attribute)
    print(attribute, "=", round(gain, 4))