# MNIST MLP Classifier

A deep learning project for **handwritten digit classification on the MNIST dataset** using a fully connected Multi-Layer Perceptron (MLP) implemented with TensorFlow and Keras.

The project covers the complete basic neural-network workflow:

- MNIST dataset loading
- Image normalization
- Image flattening
- One-hot label encoding
- Neural-network definition
- Model training
- Training and validation monitoring
- Prediction on unseen test images
- Visual inspection of classification results

---

## Overview

MNIST is one of the most widely used benchmark datasets for image classification.

It contains grayscale images of handwritten digits belonging to ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Each image has a resolution of:

```text
28 × 28 pixels
```

which corresponds to:

```text
784 input features
```

after flattening.

The goal of this project is to train a neural network to predict the digit represented in each image.

---

## Dataset

The dataset is loaded directly from TensorFlow/Keras:

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

The original dataset contains:

| Split | Images |
|---|---:|
| Training | 60,000 |
| Test | 10,000 |

Image shape:

```text
28 × 28
```

---

## Data Preprocessing

### Normalization

Pixel values originally lie in:

```text
[0, 255]
```

They are normalized using:

```python
X_train = X_train / 255
X_test = X_test / 255
```

resulting in values in:

```text
[0, 1]
```

---

### Flattening

Because the model uses fully connected Dense layers rather than convolutional layers, each image is converted from:

```text
28 × 28
```

to:

```text
784
```

features:

```python
X_train_flatten = X_train.reshape(len(X_train), 28 * 28)
X_test_flatten = X_test.reshape(len(X_test), 28 * 28)
```

---

### One-Hot Encoding

The ten digit classes are transformed into one-hot encoded vectors.

Example:

```text
Digit 3
   ↓
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

using:

```python
from tensorflow.keras.utils import to_categorical
```

---

# Neural Network Architecture

The classifier is implemented using the Keras Sequential API.

```python
model = Sequential([
    Dense(512, activation='leaky_relu', input_shape=(784,)),
    Dense(256, activation='relu'),
    Dense(20, activation='sigmoid'),
    Dense(10, activation='softmax')
])
```

The architecture is:

```text
MNIST Image
    │
    ▼
Flattened 784-dimensional vector
    │
    ▼
Dense 512
Leaky ReLU
    │
    ▼
Dense 256
ReLU
    │
    ▼
Dense 20
Sigmoid
    │
    ▼
Dense 10
Softmax
    │
    ▼
Digit probabilities
```

---

## Model Size

The network contains:

```text
538,598 trainable parameters
```

approximately distributed as:

| Layer | Parameters |
|---|---:|
| Dense 512 | 401,920 |
| Dense 256 | 131,328 |
| Dense 20 | 5,140 |
| Output Dense 10 | 210 |
| **Total** | **538,598** |

---

# Output Layer

The final layer contains:

```text
10 neurons
```

corresponding to the ten MNIST classes.

The activation function is:

```text
Softmax
```

which produces a probability distribution:

```text
P(0), P(1), P(2), ..., P(9)
```

The predicted class is obtained using:

```python
np.argmax(y_pred, axis=1)
```

---

# Training

The model is compiled using:

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Training configuration:

```text
Optimizer:       Adam
Learning rate:   0.001
Loss:            Categorical Crossentropy
Epochs:          40
Batch size:      1024
Validation:      20% of training data
```

Training is performed with:

```python
history = model.fit(
    X_train_flatten,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=1024
)
```

---

# Training Results

The model learns the MNIST classification task rapidly.

At the final epoch, the notebook reports approximately:

```text
Training accuracy:    99.96%
Validation accuracy:  97.89%

Training loss:        0.0105
Validation loss:      0.0885
```

The highest recorded validation accuracy during the run is approximately:

```text
97.99%
```

This indicates strong generalization despite the simplicity of using only fully connected layers.

---

# Learning Curves

The project visualizes both:

```text
Training loss
Validation loss
```

and:

```text
Training accuracy
Validation accuracy
```

across the 40 epochs.

These plots make it possible to examine:

- Convergence
- Generalization
- Training stability
- Potential overfitting

The increasing gap between training and validation accuracy near the end of training also provides an opportunity to discuss overfitting.

---

# Prediction

After training, predictions are generated for the complete MNIST test set:

```python
y_pred = model.predict(X_test_flatten)
```

The predicted digit is obtained using:

```python
y_pred_classes = np.argmax(y_pred, axis=1)
```

---

# Prediction Visualization

The notebook includes a helper function:

```python
show_examples()
```

that displays MNIST images together with:

```text
True label
Predicted label
```

This makes it possible to visually inspect the classifier's behavior on unseen test images.

Example:

```text
MNIST image

True:      7
Predicted: 7
```

The notebook visualizes 25 test examples after training.

---

# Project Structure

```text
mnist-mlp-classifier/
│
├── PerceptronImagenesYix.ipynb
├── README.md
└── .gitignore
```

---

# Installation

Create a Python environment and install the required dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

---

# Running the Project

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
PerceptronImagenesYix.ipynb
```

and run the cells sequentially.

The MNIST dataset is automatically downloaded by TensorFlow/Keras if it is not already available locally.

---

# Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Machine Learning Concepts

This project demonstrates:

- Supervised learning
- Multi-class classification
- Artificial neural networks
- Multi-Layer Perceptrons
- Dense neural layers
- Activation functions
- Softmax classification
- One-hot encoding
- Cross-entropy loss
- Adam optimization
- Training / validation splitting
- Image preprocessing
- Model evaluation
- Prediction visualization

---

# MLP vs. Single-Layer Perceptron

Despite the original notebook filename, the implemented classifier is more accurately described as a **Multi-Layer Perceptron (MLP)**.

A traditional perceptron contains a single trainable layer.

This project instead uses several hidden layers:

```text
784 → 512 → 256 → 20 → 10
```

with nonlinear activation functions.

These hidden layers allow the network to learn nonlinear representations of handwritten digits.

---

# Limitations

The model processes each image as a flattened 784-dimensional vector.

As a result, it does not explicitly exploit the two-dimensional spatial structure of the image.

A Convolutional Neural Network (CNN) would normally be better suited to image-classification tasks because convolutional filters can learn local visual patterns such as:

- Edges
- Curves
- Corners
- Digit strokes

The project therefore provides a useful baseline before moving to convolutional architectures.

---

# Possible Extensions

Potential improvements include:

- Convolutional Neural Networks
- Dropout regularization
- Batch normalization
- Early stopping
- Learning-rate scheduling
- Confusion matrix
- Precision and recall per digit
- Test-set accuracy calculation
- Misclassification analysis
- Data augmentation
- Hyperparameter optimization
- Comparison with CNN architectures

---

# Attribution

The notebook includes attribution to **Guillermo Iglesias (Universidad Politécnica de Madrid)**.

The original material indicates a **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)** license.

Any redistribution or modification should preserve the corresponding attribution and comply with the terms of the original license.

---

# Disclaimer

This project is intended for educational and experimental purposes.

Results may vary depending on TensorFlow version, hardware, random initialization, and training configuration.
