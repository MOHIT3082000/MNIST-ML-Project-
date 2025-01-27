# MNIST-ML-Project-
This project classifies handwritten digits from the MNIST dataset using a Keras deep learning model. It preprocesses data, visualizes samples, builds a Sequential model with Dense and ReLU layers, and evaluates performance using accuracy and confusion matrix. Custom images can also be predicted using the trained model.
MNIST Digit Classification with Deep Learning
This project focuses on classifying handwritten digits from the MNIST dataset using deep learning techniques. The dataset is loaded from keras.datasets and consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The goal is to build a robust model capable of accurately predicting these digits.

Project Overview:
Data Preprocessing:

The MNIST dataset is preprocessed using numpy and pandas for efficient handling and manipulation.
Data is normalized to the range [0, 1] and reshaped to fit the model’s input layer.
train_test_split is used to split the data into training and testing sets, and their shapes are validated to ensure correct format.
Data Visualization:

Various visualizations are generated using seaborn and matplotlib.pyplot to better understand the data distribution and characteristics.
Sample images from the dataset are displayed to provide a clear view of the problem domain.
Model Architecture:

A Keras Sequential model is built with:
Flatten layer to convert the 2D image arrays into 1D vectors.
Dense layers with ReLU activation for non-linearity and Sigmoid activation at the output layer for binary classification.
The model is designed to learn the patterns of handwritten digits and output a probability distribution over the possible digits (0-9).
Model Training:

The model is compiled with the Adam optimizer and binary cross-entropy loss function.
It is trained using the training set (x_train, y_train) for a specified number of epochs to optimize the weights.
Model Evaluation:

Accuracy is checked on the test data using x_test, y_test.
The predictions are evaluated using a Confusion Matrix to assess the model's performance across different digits.
Image Prediction:

The project also allows for custom image input, where users can input their own handwritten digit images, which are then processed and classified by the trained model.
Libraries Used:
TensorFlow & Keras: For building and training the deep learning model.
NumPy & Pandas: For efficient data manipulation and preprocessing.
Matplotlib & Seaborn: For data visualization.
OpenCV: For reading and processing custom images for prediction.
Confusion Matrix: For evaluating the model’s performance on test data.
