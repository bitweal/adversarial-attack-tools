from models.mobilenet_v2 import mobilenet_v2
from models.inception_v3 import inception_v3
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

    @staticmethod
    def load_labels(classes_path):
        with open(classes_path, 'r') as f:
            labels = [line.strip() for line in f]
        return labels

    @staticmethod
    def save_tensor_image(image_tensor, path_save_image):
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        processed_image = transforms.ToPILImage()(image_tensor)
        processed_image.save(f'media/{path_save_image}.jpg')

    @staticmethod
    def save_image(image, path_save_image):
        processed_image = image.resize((224, 224), resample=Image.BILINEAR)
        processed_image.save(f'media/{path_save_image}.jpg')

    def load_image(self, image_path):
        self.image = Image.open(image_path)
        self.image_in_tensor = transforms.ToTensor()(self.image).unsqueeze(0)

    def tensor_to_batch(self, input_tensor):
        self.batch = input_tensor.unsqueeze(0)
        return self.batch

    def compute_gradient(self):
        batch = copy.deepcopy(self.batch)
        model = copy.deepcopy(self.model)
        batch.requires_grad = True
        output = model(batch)
        predicted_class = output.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(output, predicted_class)
        model.zero_grad()
        loss.backward()
        data_grad = batch.grad.data
        self.data_grad = data_grad

    def bim_attack(self):
        self.compute_gradient()
        predicted_class = None
        data_grad = copy.deepcopy(self.data_grad)
        sign_data_grad = data_grad.sign()
        image = copy.deepcopy(self.image)
        image_in_tensor = copy.deepcopy(self.image_in_tensor)

        for eps in np.arange(0, 0.5, 0.01):
            perturbed_image = self.image_in_tensor + eps * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            self.save_tensor_image(perturbed_image, f'bim_attack{eps}')
            print(eps)
            self.load_image(f'media/bim_attack{eps}.jpg')

            if self.predicted_class is None:
                self.predicted_class = self.predict()
            else:
                predicted_class = self.predict()
                self.compute_gradient()
                data_grad = copy.deepcopy(self.data_grad)
                sign_data_grad = data_grad.sign()

            if predicted_class != self.predicted_class:
                break

        self.image = image
        self.image_in_tensor = image_in_tensor

    def predict(self):
        input_image = self.image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = self.tensor_to_batch(input_tensor)
        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = self.class_labels[predicted_class_idx]

        print("Predicted class:", predicted_class)
        print("Probability:", probabilities[predicted_class_idx].item())

        if self.predicted_class is None:
            self.predicted_class = predicted_class
        return predicted_class


if __name__ == "__main__":
    model = mobilenet_v2(pretrained=True)
    model.eval()
    filename = 'media/dog.jpg'
    file_classes = 'imagenet_classes.txt'
    attack = AdversarialAttack(model, file_classes)
    attack.load_image(filename)
    attack.predict()
    attack.bim_attack()
