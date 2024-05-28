# Adversarial Attack Tools

Welcome to the Adversarial Attack Tools repository! This repository provides Python code to conduct adversarial attacks on image classification models. It offers an easy-to-use platform for experimenting with adversarial attacks using pre-trained models such as MobileNetV2 and InceptionV3.

## Purpose

The purpose of this repository is to facilitate experimentation with adversarial attacks within the domain of image classification. Adversarial attacks involve subtly modifying input data to mislead machine learning models, leading them to make incorrect predictions. Understanding and addressing these vulnerabilities are crucial for enhancing the security and robustness of machine learning systems.

## Features

- **Adversarial Attack Implementation**: The repository implements the Basic Iterative Method (BIM) and Fast Gradient Sign Method (FGSM) for generating adversarial examples.
- **Supported Models**: Pre-trained MobileNetV2 and Inception v3 model. This models is widely used for image classification tasks and serves as the target for adversarial attacks.
- **User-Friendly Interface**: The code is structured for ease of understanding and use. It includes functionalities for loading images, making predictions, and conducting adversarial attacks.

## How to Use

1. **Clone the Repository**: Begin by cloning this repository to your local machine. You can achieve this by running `git clone https://github.com/bitweal/adversarial-attack-tools.git` in your terminal or command prompt.

2. **Install Dependencies**: Ensure Python is installed along with the required libraries specified in `requirements.txt`. You can install them via `pip install -r requirements.txt`.

3. **Prepare the Image**: Place the image intended for the attack in the `media` folder. Ensure it's in a supported format.

4. **Run the Code**: Execute the `adversarial_attack.py` file. This action will load the pre-trained model, make predictions on the input image, and conduct an adversarial attack using the BIM method.

5. **Interpret the Results**: Upon running the code, you'll find the adversarially perturbed images saved in the `media` folder. Each image will be labeled with the epsilon value used for the attack. Additionally, the predicted class and probability for each image will be printed in the console.

## Repository Structure

- **models/**: Directory containing model definitions and pre-trained weights.
  - **mobilenet_v2.py**: Contains the definition of the MobileNetV2 model.
  - **inception_v3.py**: Contains the definition of the InceptionV3 model.
- **adversarial_attack.py**: Implements the AdversarialAttack class for crafting adversarial examples using the BIM and FGSM. Entry point of the application. It loads the pre-trained model, performs predictions, and conducts adversarial attacks.
- **imagenet_classes.txt**: Contains the class labels for the ImageNet dataset.
- **media/**: Directory to store input images and adversarially perturbed images.
