from models.mobilenet_v2 import mobilenet_v2
from models.inception_v3 import inception_v3
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, Inception_V3_Weights
import errors
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import copy
import numpy as np


class AdversarialAttack:
    def __init__(self, model, classes_path):
        self.model = model
        self.class_labels = self.load_labels(classes_path)
        self.image = None
        self.image_in_tensor = None
        self.batch = None
        self.data_grad = None
        self.predicted_class = None
        self.features = None

    @staticmethod
    def load_labels(classes_path):
        with open(classes_path, 'r') as f:
            labels = [line.strip() for line in f]
        return labels

    @staticmethod
    def save_tensor_to_image(image_tensor, path_save_image):
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        processed_image = transforms.ToPILImage()(image_tensor)
        processed_image.save(f'media/{path_save_image}.jpg')

    @staticmethod
    def save_image(image, path_save_image):
        processed_image = image.resize((224, 224), resample=Image.BILINEAR)
        processed_image.save(f'media/{path_save_image}.jpg')

    def load_image(self, image_path):
        self.image = Image.open(image_path).convert("RGB")
        self.image_in_tensor = transforms.ToTensor()(self.image)
        self.batch = self.image_in_tensor.unsqueeze(0)

    def compute_gradient(self):
        self.batch.requires_grad = True
        output = self.model(self.batch)
        predicted_class = output.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(output, predicted_class)
        self.model.zero_grad()
        loss.backward()
        self.data_grad = self.batch.grad.data

    def fgsm_attack(self, dynamic_epsilon=True, epsilon=0.01, size_step_epsilon=0.01):
        if self.image is None:
            raise errors.ImageException('self.image was not loaded, use load_image()')

        image = copy.deepcopy(self.image)
        image_in_tensor = copy.deepcopy(self.image_in_tensor)
        image_batch = copy.deepcopy(self.batch)

        if self.predicted_class is None:
            self.predicted_class = self.predict()

        max_steps = 500
        for step in range(max_steps):
            print('step: ', step)
            print('epsilon: ', epsilon)
            self.compute_gradient()
            perturbed_image = self.image_in_tensor + epsilon * self.data_grad.sign()
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            name_new_image = f'{self.model.__class__.__name__}_fgsm_attack_{step}'
            self.save_tensor_to_image(perturbed_image, name_new_image)
            self.load_image(f'media/{name_new_image}.jpg')
            predicted_class = self.predict()

            if dynamic_epsilon:
                epsilon += size_step_epsilon

            if predicted_class != self.predicted_class and predicted_class:
                break

        self.image = image
        self.image_in_tensor = image_in_tensor
        self.batch = image_batch

    def bim_attack(self, dynamic_epsilon=True, epsilon=0.01, size_step_epsilon=0.01):
        if self.image is None:
            raise errors.ImageException('self.image was not loaded, use load_image()')

        image = copy.deepcopy(self.image)
        image_in_tensor = copy.deepcopy(self.image_in_tensor)
        image_batch = copy.deepcopy(self.batch)

        if self.predicted_class is None:
            self.predicted_class = self.predict()

        max_steps = 1000
        for step in range(max_steps):
            print('step: ', step)
            print('epsilon: ', epsilon)
            self.compute_gradient()
            perturbed_image = self.image_in_tensor + epsilon * self.data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            name_new_image = f'{self.model.__class__.__name__}_bim_attack_{step}'
            self.save_tensor_to_image(perturbed_image, name_new_image)
            self.load_image(f'media/{name_new_image}.jpg')
            predicted_class = self.predict()

            if dynamic_epsilon:
                epsilon += size_step_epsilon

            if predicted_class != self.predicted_class and predicted_class:
                break

        self.image = image
        self.image_in_tensor = image_in_tensor
        self.batch = image_batch

    def prediction(self, image, internal=None):
        layers = []
        prediction = self.model(image)

        if internal is None:
            return layers, prediction

        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in internal:
                layers.append(image)
        return layers, prediction

    def dispersion_reduction(self, epsilon=0.01, alpha=0.001, layer_idx=-1, internal_layers=None):
        if self.image is None:
            raise errors.ImageException('self.image was not loaded, use load_image()')

        image = copy.deepcopy(self.image)
        image_in_tensor = copy.deepcopy(self.image_in_tensor)
        image_batch = copy.deepcopy(self.batch)

        if self.predicted_class is None:
            self.predicted_class = self.predict()

        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).eval()
        perturbed_image = copy.deepcopy(self.batch)
        max_steps = 1000
        for step in range(max_steps):
            print('step: ', step)

            self.batch.requires_grad = True
            internal_features, output = self.prediction(self.batch, internal_layers)
            logit = internal_features[layer_idx]
            loss = -1 * logit.std()
            self.model.zero_grad()
            loss.backward()
            data_grad = self.batch.grad.data

            perturbed_image = perturbed_image + alpha * data_grad.sign()
            perturbed_image = torch.clamp(perturbed_image, image_batch - epsilon, image_batch + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            name_new_image = f'{self.model.__class__.__name__}_dispersion_reduction_{step}'
            self.save_tensor_to_image(perturbed_image, name_new_image)
            self.load_image(f'media/{name_new_image}.jpg')
            predicted_class = self.predict()

            if predicted_class != self.predicted_class and predicted_class:
                break

        self.image = image
        self.image_in_tensor = image_in_tensor
        self.batch = image_batch

    def dispersion_amplification(self):
        pass

    def predict(self):
        if self.image is None:
            raise errors.ImageException('self.image was not loaded, use load_image()')

        with torch.no_grad():
            output = self.model(self.batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = self.class_labels[predicted_class_idx]

        print("Predicted class:", predicted_class)
        print("Probability:", probabilities[predicted_class_idx].item())

        if self.predicted_class is None:
            self.predicted_class = predicted_class
        return predicted_class


if __name__ == "__main__":
    model_mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    model_mobilenet_v2.eval()
    filename = 'media/dog.jpg'
    file_classes = 'imagenet_classes.txt'
    list_index_layers = [i for i in range(len(model_mobilenet_v2.features))]
    attack_layer_idx = 4

    attack = AdversarialAttack(model_mobilenet_v2, file_classes)
    attack.load_image(filename)
    #attack.predict()
    #attack.fgsm_attack(True, 0.01, 0.01)
    #attack.bim_attack(True, 0.01, 0.01)

    attack.dispersion_reduction(0.1, 0.004, attack_layer_idx, list_index_layers)
    #attack.dispersion_amplification()
