import numpy as np
import gzip
import urllib.request
from struct import unpack

# Load MNIST data
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(16)
        num_images = unpack('>I', f.read(4))[0]
        num_rows = unpack('>I', f.read(4))[0]
        num_cols = unpack('>I', f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        return images / 255.0

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(8)
        num_labels = unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Download MNIST dataset
def download_mnist():
    urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    for key, url in urls.items():
        response = urllib.request.urlopen(url)
        with open(f'{key}.gz', 'wb') as f:
            f.write(response.read())

download_mnist()


# Load dataset
train_images = load_mnist_images('train_images.gz')
train_labels = load_mnist_labels('train_labels.gz')
test_images = load_mnist_images('t10k-images.gz')
test_labels = load_mnist_labels('t10k-labels.gz')

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

train_labels_one_hot = one_hot_encode(train_labels, 10)
test_labels_one_hot = one_hot_encode(test_labels, 10)

input_size = 28 * 28 
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 10
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a1, a2

def backward_pass(X, y_true, a1, a2):
    m = X.shape[0]
    dZ2 = a2 - y_true
    dW2 = np.dot(a1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(np.dot(X, W1) + b1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def train(X, y):
    global W1, b1, W2, b2
    for epoch in range(epochs):
        a1, a2 = forward_pass(X)
        dW1, db1, dW2, db2 = backward_pass(X, y, a1, a2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        predictions = np.argmax(a2, axis=1)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy * 100:.2f}%')

def predict(X):
    _, a2 = forward_pass(X)
    return np.argmax(a2, axis=1)

# Train the model
train(train_images, train_labels_one_hot)

# Test the model
test_predictions = predict(test_images)
accuracy = np.mean(test_predictions == test_labels)
print(f'Test accuracy: {accuracy * 100:.2f}%')