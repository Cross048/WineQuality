import logging
import math
import warnings
import numpy as np
import pandas as pd
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET

# Load red and white wine data
url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Load the datasets
red_wine = pd.read_csv(url_red, sep=';')
white_wine = pd.read_csv(url_white, sep=';')

# Label the wines (0 for red, 1 for white)
red_wine['label'] = 0
white_wine['label'] = 1

# Combine the datasets
wine_data = pd.concat([red_wine, white_wine], axis=0)

# Separate features and labels
X = wine_data.drop('label', axis=1).values
y = wine_data['label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Suppress Numba low-performance warnings
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
logging.getLogger('numba').setLevel(logging.ERROR)


# Perceptron class using CUDA
class PerceptronCUDA:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000000):
        self.weights = np.zeros(input_size + 1, dtype=np.float32)  # Weights initialized to zero (including bias)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    @staticmethod
    @cuda.jit
    def predict_kernel(X, weights, predictions):
        idx = cuda.grid(1)
        if idx < X.shape[0]:
            linear_output = weights[0]  # Bias
            for i in range(X.shape[1]):
                linear_output += X[idx, i] * weights[i + 1]
            predictions[idx] = 1 if linear_output >= 0 else 0

    @staticmethod
    @cuda.jit
    def fit_kernel(X, y, weights, learning_rate):
        idx = cuda.grid(1)
        if idx < X.shape[0]:
            prediction = 0
            linear_output = weights[0]  # Bias
            for i in range(X.shape[1]):
                linear_output += X[idx, i] * weights[i + 1]
            prediction = 1 if linear_output >= 0 else 0
            update = learning_rate * (y[idx] - prediction)
            for i in range(X.shape[1]):
                weights[i + 1] += update * X[idx, i]
            weights[0] += update

    def fit(self, X, y):
        # Allocate memory for device variables
        X_device = cuda.to_device(X)
        y_device = cuda.to_device(y)
        weights_device = cuda.to_device(self.weights)

        # Parallelize training with CUDA
        threads_per_block = 256  # You can increase this depending on your GPU's capabilities
        blocks_per_grid = math.ceil(X.shape[0] / threads_per_block)

        for epoch in range(self.epochs):
            self.fit_kernel[blocks_per_grid, threads_per_block](X_device, y_device, weights_device, self.learning_rate)

        # Copy back weights from device to host
        self.weights = weights_device.copy_to_host()

    def predict(self, X):
        # Allocate memory for device variables
        X_device = cuda.to_device(X)
        predictions_device = cuda.device_array(X.shape[0], dtype=np.int32)

        # Parallelize prediction with CUDA
        threads_per_block = 1024
        blocks_per_grid = math.ceil(X.shape[0] / threads_per_block)
        self.predict_kernel[blocks_per_grid, threads_per_block](X_device, self.weights, predictions_device)

        # Copy predictions back to host
        return predictions_device.copy_to_host()

    def save_weights_to_xml(self, filename="weights.xml"):
        # Create XML structure
        root = ET.Element("Perceptron")
        weights_elem = ET.SubElement(root, "Weights")

        for i, weight in enumerate(self.weights):
            weight_elem = ET.SubElement(weights_elem, "Weight", index=str(i))
            weight_elem.text = str(weight)

        # Create and write to XML file
        tree = ET.ElementTree(root)
        tree.write(filename)


# Train the model using PerceptronCUDA
input_size = X_train.shape[1]
perceptron_cuda = PerceptronCUDA(input_size=input_size, learning_rate=0.01, epochs=1000)
perceptron_cuda.fit(X_train, y_train)

# Evaluate the model
predictions = perceptron_cuda.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy:.2f}')

# Count red and white wines
num_red_wines = (y == 0).sum()
num_white_wines = (y == 1).sum()

# Display the counts
print(f'Total number of red wines: {num_red_wines}')
print(f'Total number of white wines: {num_white_wines}')

# Save the weights to an XML file
perceptron_cuda.save_weights_to_xml("weights.xml")
print("Weights saved to 'weights.xml'")
