# MNIST Handwritten Digit Recognition Web Application

This is a simple web application that recognizes handwritten digits using a pre-trained Convolutional Neural Network (CNN) model. Users can upload an image of a handwritten digit, and the application will predict the digit and display the result.

## Code Explanation[YouTube Video]
<a href="https://youtu.be/2GQm10JJ1BY"><img src="https://github.com/bipin-saha/MNIST_HWDR_WebApp/blob/main/Thumbnail.jpg" width="35%" height="35%"></img></a>



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [License](#license)

## Installation
To run this web application locally, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/bipin-saha/MNIST_HWDR_WebApp.git
    ```

2. Install the required dependencies. You can use a virtual environment to manage the dependencies:

    ```bash
    cd MNIST_HWDR_WebApp
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Download the pre-trained model checkpoint file (e.g., MNIST_HWDR_2D-feedforward.pth) and place it in the `models` folder.

4. Run the Flask application:

    ```bash
    python app.py
    ```

5. Open a web browser and go to http://localhost:5000 to use the application.

## Usage
Access the web application by visiting http://localhost:5000 in your web browser.

1. Click the "Choose File" button to upload an image containing a handwritten digit.
2. Click the "Predict" button to initiate the digit recognition process.
3. The predicted digit will be displayed on the web page along with the uploaded image.
4. You can upload more images to make additional predictions.

## Model Training

If you want to train the MNIST digit recognition model yourself, you can follow these steps:

### Model Architecture

The MNIST Digit Recognition model is a fundamental deep-learning model used for recognizing handwritten digits. Trained on the MNIST dataset, it typically employs convolutional neural networks (CNNs) to analyze and classify grayscale images of handwritten digits, each ranging from 0 to 9. This model plays a crucial role in introductory machine learning and computer vision, serving as a benchmark for evaluating the performance of various image classification algorithms. It's widely used for educational purposes and serves as a foundation for more complex image recognition tasks.
Here's a brief overview of its architecture:

- **Input Layer:** Accepts grayscale images of size 28x28 pixels.
- **Convolutional Layer 1 (conv1):** Applies 32 filters of size 5x5 with ReLU activation and same padding. This layer reduces the spatial dimensions.
- **Max-Pooling 1:** Performs max-pooling with a 2x2 kernel to further reduce the spatial dimensions.
- **Convolutional Layer 2 (conv2):** Applies 64 filters of size 5x5 with ReLU activation and same padding.
- **Max-Pooling 2:** Another max-pooling layer to further reduce spatial dimensions.
- **Fully Connected Layer 1 (fc1):** Contains 1024 neurons with ReLU activation. It takes the flattened output of the previous layers.
- **Dropout Layer:** Applies dropout during training to prevent overfitting.
- **Fully Connected Layer 2 (fc2):** The final layer with 10 neurons (one for each digit class), followed by a log-softmax activation.

### Training Script

Open the `training/MNIST_CNN_FeedFrward.ipynb` file in Google Colab/Kaggle. This script is responsible for training the `MnistModel` using the MNIST dataset. You can modify hyperparameters like batch size, learning rate, and the number of training epochs if needed.

## Folder Structure
- `MNIST_HWDR_WebApp/` - Root directory of the application.
- `app.py` - The Flask web application script.
- `training/MNIST_CNN_FeedFrward.ipynb` - Script for training the model in Colab/Kaggle (if needed).
- `uploads/` - Folder where uploaded images are stored.
- `models/` - Folder for storing pre-trained model checkpoint files.
-  `samples/` - Sample images for testing web app (not limited too).
- `static/` - Folder for static assets (e.g., CSS, JavaScript).
- `templates/` - HTML templates for rendering web pages.
-  `uploads/` - Uploaded photo.
- `requirements.txt` - List of Python dependencies.

## Technologies Used
- Python
- Flask
- PyTorch (for FeedForward CNN)
- HTML/CSS
- PIL (Python Imaging Library) for image processing
