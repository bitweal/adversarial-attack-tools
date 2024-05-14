from mobilenet_v2 import mobilenet_v2
from inception_v3 import inception_v3
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class DispersionAttack:
    def __init__(self, model, image_path, classes_path):
        self.model = model
        self.image_path = image_path
        self.classes_path = classes_path
        self.class_labels = self.load_labels()

    @staticmethod
    def save_tensor_image(self, image_tensor, path_save_image):
        processed_image = transforms.ToPILImage()(image_tensor)
        processed_image.save(f'media/{path_save_image}.jpg')

    @staticmethod
    def save_image(self, image, path_save_image):
        processed_image = image.resize((224, 224), resample=Image.BILINEAR)
        processed_image.save(f'media/{path_save_image}.jpg')

    def load_labels(self):
        with open(self.classes_path, 'r') as f:
            labels = [line.strip() for line in f]
        return labels

    def predict(self):
        input_image = Image.open(self.image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = self.class_labels[predicted_class_idx]
        print("Predicted class:", predicted_class)
        print("Probability:", probabilities[predicted_class_idx].item())


if __name__ == "__main__":
    model = mobilenet_v2(pretrained=True)
    model.eval()
    filename = 'media/dog.jpg'
    file_classes = 'imagenet_classes.txt'
    attack = DispersionAttack(model, filename, file_classes)
    attack.predict()
