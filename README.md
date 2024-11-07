
# Handwritten Digit Recognition

This Jupyter Notebook provides an implementation for recognizing handwritten digits using a neural network model trained on the MNIST dataset. The model is built with Keras and TensorFlow, using a simple architecture with two hidden layers.

## Dataset
The notebook uses the **MNIST dataset**, which contains 28x28 pixel grayscale images of handwritten digits ranging from 0 to 9. The dataset is loaded directly from `tensorflow.keras.datasets.mnist`, which provides separate training and testing sets.

## Model Architecture
The neural network model is constructed with the following layers:
- **Input Layer**: Accepts 784 input features (flattened 28x28 images).
- **Hidden Layer 1**: Dense layer with 128 neurons and ReLU activation.
- **Hidden Layer 2**: Dense layer with 64 neurons and ReLU activation.
- **Output Layer**: Dense layer with 10 neurons and softmax activation for multi-class classification (digits 0 through 9).

## Compilation
The model is compiled with:
- **Loss Function**: Categorical cross-entropy, suitable for multi-class classification.
- **Optimizer**: Adam optimizer, a popular choice for training deep learning models.
- **Metric**: Accuracy, to evaluate performance on the test set.

## Training
The model is trained on the MNIST training set for 5 epochs with a batch size of 32.

## Evaluation
After training, the model's performance is evaluated on the test set, with accuracy as the primary metric.

## Predictions
The notebook includes code to make predictions on a sample of test images and compares them to the actual labels.

## Requirements
- Python 3.x
- Jupyter Notebook
- TensorFlow and Keras
- NumPy and Matplotlib (for data processing and visualization)

## Running the Notebook
1. Open the notebook in Jupyter.
2. Run each cell sequentially to load the dataset, build and train the model, and evaluate its performance.

## Example Usage
The notebook demonstrates the complete workflow for handwritten digit recognition, from loading the dataset and training the model to evaluating its accuracy and making sample predictions.

## Acknowledgments
This project uses the MNIST dataset, which is a standard benchmark dataset for image classification.
