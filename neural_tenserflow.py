import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          
    layers.Dense(128, activation='relu'),        
    layers.Dropout(0.2),                          
    layers.Dense(10, activation='softmax')        
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
predictions = model.predict(test_images)

def plot_image(image, label, prediction):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(f"True label: {label}, Prediction: {np.argmax(prediction)}")
    plt.axis('off')
    plt.show()

for i in range(5):
    plot_image(test_images[i], test_labels[i], predictions[i])
