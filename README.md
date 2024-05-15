# Adversarial Attack Repository

Welcome to the Adversarial Attack Repository! This repository contains Python code to perform adversarial attacks on image classification models. It is designed to be user-friendly and allows anyone to experiment with adversarial attacks using pre-trained models like MobileNetV2 and InceptionV3.

## Purpose

The purpose of this repository is to provide a platform for experimenting with adversarial attacks in the context of image classification. Adversarial attacks are a method of subtly modifying input data to mislead machine learning models, causing them to make incorrect predictions. This has implications for the security and robustness of machine learning systems, making it essential to explore their vulnerabilities through tools like this repository.

## Features

- **Adversarial Attack Implementation**: The repository implements the Basic Iterative Method (BIM) for generating adversarial examples. 
- **Supported Models**: Pre-trained model MobileNetV2. This model is widely used for image classification tasks and serves as a target for adversarial attacks.
- **User-Friendly Interface**: The code is structured to be easy to understand and use. It includes functions for loading images, making predictions, and conducting adversarial attacks.

## How to Use

1. **Clone the Repository**: Start by cloning this repository to your local machine. You can do this by running `git clone https://github.com/bitweal/improvement-of-the-dispersion-attack-method.git` in your terminal or command prompt.

2. **Install Dependencies**: Ensure you have Python installed along with the required libraries specified in `requirements.txt`. You can install them using `pip install -r requirements.txt`.

3. **Prepare the Image**: Place the image you want to use for the attack in the `media` folder. Ensure it's in a supported format

4. **Run the Code**: Execute the `adversarial_attack.py` file. This will load the pre-trained model, perform predictions on the input image, and then conduct an adversarial attack using the BIM method.

5. **Interpret the Results**: After running the code, you'll find the adversarially perturbed images saved in the `media` folder. Each image will be labeled with the epsilon value used for the attack. Additionally, the predicted class and probability for each image will be printed in the console.

## Repository Structure

- **mobilenet_v2.py**: Contains the definition of the MobileNetV2 model.
- **inception_v3.py**: Contains the definition of the InceptionV3 model.
- **adversarial_attack.py**: Implements the AdversarialAttack class for crafting adversarial examples using the BIM method. Entry point of the application. Loads the pre-trained model, performs predictions, and conducts adversarial attacks.
- **imagenet_classes.txt**: Contains the class labels for the ImageNet dataset.
- **media/**: Directory to store input images and adversarially perturbed images.

