import json
import os

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from src.model import Net


@st.cache_resource
def load_model(num_classes: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(num_classes=num_classes)
    net.to(device=device)

    return net


class Predictor:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def load_list_models(self):
        list_models = []
        for file_name in os.listdir('models'):
            if file_name.endswith(".pth"):
                list_models.append(file_name)

        return list_models


    def load_classes(self, name_model):

        name_model = name_model.replace(".pth", "")

        with open(f"models/classes_{name_model}.json", "r") as file:
            classes = json.load(file)
        return classes

    def predict(self, name_model, image):

        if image is None:
            print("No image file")
            return

        image = Image.open(image)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image = transform(image)
        image = image.unsqueeze(0)

        image = image.to(device=self.device)

        classes = self.load_classes(name_model=name_model)

        model = load_model(num_classes=len(classes))

        model.load_state_dict(torch.load(f"models/{name_model}"))

        model.eval()

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            percentage = probabilities * 100

        values = percentage[0].tolist()

        results = dict(zip(classes, values))
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        return results


