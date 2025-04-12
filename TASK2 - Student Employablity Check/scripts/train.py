# train.py
import sys
import os
# Add parent directory to Python path so that model.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from model import Perceptron  # Import our custom model

# 1. Read the dataset (adjust the path as needed)
df = pd.read_csv(r"data\data.csv")
df.dropna(subset=["CLASS"], inplace=True)  # Remove rows with unrecognized class labels

# 2. Drop "Name of Student" (non-predictive identifier)
df.drop(columns=["Name of Student"], inplace=True)

# 3. Encode "CLASS" into numeric form: "Employable" -> 1, "LessEmployable" -> 0
df["CLASS"] = df["CLASS"].map({"Employable": 1, "LessEmployable": 0})

# 4. Define numeric columns (ratings) and impute missing values
rating_cols = [
    "GENERAL APPEARANCE",
    "MANNER OF SPEAKING",
    "PHYSICAL CONDITION",
    "MENTAL ALERTNESS",
    "SELF-CONFIDENCE",
    "ABILITY TO PRESENT IDEAS",
    "COMMUNICATION SKILLS"
]
imputer = SimpleImputer(strategy="mean")
df[rating_cols] = imputer.fit_transform(df[rating_cols])

# 5. Outlier Treatment: clip values to the valid range [1, 5]
for col in rating_cols:
    df[col] = np.clip(df[col], 1, 5)

# 6. Feature Selection: choose the top 6 features for training
selected_features = [
    "GENERAL APPEARANCE",
    "MANNER OF SPEAKING",
    "MENTAL ALERTNESS",
    "SELF-CONFIDENCE",
    "ABILITY TO PRESENT IDEAS",
    "COMMUNICATION SKILLS"
]
X = df[selected_features]
y = df["CLASS"]

# 7. Scale the selected features for balanced training
#scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1)) 
X_scaled = scaler.fit_transform(X)

# 8. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9. Initialize and train the Perceptron
perceptron = Perceptron(input_dim=X_train.shape[1])
epochs = 1000
learning_rate = 0.1

train_losses, test_losses, train_accuracies, test_accuracies = perceptron.train(
    X_train, X_test, y_train, y_test, epochs, learning_rate
)

# 10. Evaluate the final model
def evaluate(model, X, y):
    """Compute accuracy of the model on given data."""
    predictions = model.predict(X)
    predictions = np.round(predictions)  # Convert probabilities to 0 or 1
    accuracy = np.mean(predictions == y) * 100
    return accuracy

train_accuracy = evaluate(perceptron, X_train, y_train)
test_accuracy = evaluate(perceptron, X_test, y_test)
print(f"Final Training Accuracy: {train_accuracy:.2f}%")
print(f"Final Testing Accuracy: {test_accuracy:.2f}%")

# 11. Save the model with Pickle (save to the models/ folder)
with open('models\perp.pkl', 'wb') as file: 
    pickle.dump(perceptron, file)

# 12. Plot the Training Results (Loss and Accuracy over epochs)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss over Epochs
axes[0].plot(train_losses, label="Training Loss")
axes[0].plot(test_losses, label="Testing Loss")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Plot Accuracy over Epochs
axes[1].plot(train_accuracies, label="Training Accuracy")
axes[1].plot(test_accuracies, label="Testing Accuracy")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.tight_layout()
plt.show()
