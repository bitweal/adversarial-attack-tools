# Adversarial Attack Tools

Welcome to the Adversarial Attack Tools repository! This repository provides Python code to conduct adversarial attacks on image classification models. It offers an easy-to-use platform for experimenting with adversarial attacks using a variety of models available in `torchvision`.

## Purpose

The purpose of this repository is to facilitate experimentation with adversarial attacks within the domain of image classification. Adversarial attacks involve subtly modifying input data to mislead machine learning models, leading them to make incorrect predictions. Understanding and addressing these vulnerabilities are crucial for enhancing the security and robustness of machine learning systems.

## Features

- **Adversarial Attack Implementation**: The repository implements the Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM),  dispersion reduction (DR), and dispersion amplification (DA) for generating adversarial examples.
- **User-Friendly Interface**: The code is structured for ease of understanding and use. It includes functionalities for loading images, making predictions, and conducting adversarial attacks.

## How to Use

1. **Clone the Repository**: Begin by cloning this repository to your local machine. You can achieve this by running `git clone https://github.com/bitweal/adversarial-attack-tools.git` in your terminal or command prompt.

2. **Install Dependencies**: Ensure Python is installed along with the required libraries specified in `requirements.txt`. You can install them via `pip install -r requirements.txt`.

3. **Using the Code**:
    - **Load an Image**: You can load an image using the `load_image` method of the `AdversarialAttack` class.
    - **Resize an Image**: You can resize an image using the `resize_image` method, which saves the resized image in the `media` folder.
    - **Predict the Class**: Use the `predict` method to get the model's prediction for the loaded image.

**Adversarial Attacks**: The `AdversarialAttack` class provides several methods to perform adversarial attacks:\
    - **FGSM Attack**: `fgsm_attack(dynamic_epsilon, epsilon, size_step_epsilon, step_after_change_class)`\
    - **BIM Attack**: `bim_attack(dynamic_epsilon, epsilon, size_step_epsilon, step_after_change_class)`\
    - **Dispersion Reduction**: `dispersion_reduction(dynamic_alpha, alpha, size_step_alpha, attack_budget, attack_layer_idx, step_after_change_class)`\
    - **Dispersion Amplification**: `dispersion_amplification(dynamic_alpha, alpha, size_step_alpha, attack_budget, attack_layer_idx, step_after_change_class)`

## Repository Structure

- **adversarial_attack.py**: The main script to run the adversarial attacks using different models. Implements the `AdversarialAttack` class for crafting adversarial examples using the BIM, FGSM, DR adn DR methods.
- **imagenet_classes.txt**: Contains the class labels for the ImageNet dataset.
- **media/**: Directory to store input images and adversarially perturbed images.