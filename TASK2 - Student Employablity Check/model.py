# model.py
import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function (assuming x is already sigmoid output)."""
    return x * (1 - x)

class Perceptron:
    """Custom Perceptron model with a sigmoid activation."""
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = 0

    def predict(self, X):
        """Compute the model output for each sample in X."""
        linear_output = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_output)

    def train(self, X_train, X_test, y_train, y_test, epochs, learning_rate):
        """Train the perceptron using gradient descent, returning metrics each epoch."""
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            # Update weights for each training sample
            for i in range(X_train.shape[0]):
                # Forward pass
                linear_output = np.dot(X_train[i], self.weights) + self.bias
                prediction = sigmoid(linear_output)
                # Calculate error (using y_train.iloc because y_train is a pandas Series)
                error = y_train.iloc[i] - prediction
                # Update weights and bias
                self.weights += learning_rate * error * X_train[i] * sigmoid_derivative(prediction)
                self.bias += learning_rate * error * sigmoid_derivative(prediction)

            # Calculate metrics once per epoch
            # Training metrics
            train_pred = self.predict(X_train)
            train_loss = np.mean((y_train - train_pred) ** 2)
            train_losses.append(train_loss)
            train_acc = np.mean(np.round(train_pred) == y_train) * 100
            train_accuracies.append(train_acc)

            # Testing metrics
            test_pred = self.predict(X_test)
            test_loss = np.mean((y_test - test_pred) ** 2)
            test_losses.append(test_loss)
            test_acc = np.mean(np.round(test_pred) == y_test) * 100
            test_accuracies.append(test_acc)

            # Print progress every 100 epochs (adjust as needed)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        return train_losses, test_losses, train_accuracies, test_accuracies
