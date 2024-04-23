import json
import streamlit as st
from src.model import Net
from src.utils import Utils
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


class Train:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def train_model(self, epochs: int, model_name: str):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        Utils().clean_corrupt_images(root_dir="images_files")

        data_files = ImageFolder(root="images_files", transform=transform)

        classes = data_files.classes

        with open(f'models/classes_{model_name}.json', 'w') as file:
            file.write(json.dumps(classes, indent=4))

        train_size = int(0.85 * len(data_files))
        test_size = len(data_files) - train_size
        train_set, test_set = random_split(data_files, [train_size, test_size])

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)

        model = Net(num_classes=len(classes))
        model.to(device=self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        train_result = []

        progress_bar = st.progress(0, "Initializing training...")

        for epoch in range(epochs):

            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            accuracy = self._evaluate_model(model, test_loader)

            info = f'Epoch {epoch + 1} -> Loss: {round(running_loss / len(train_loader), 2)} - Accuracy: {round(accuracy, 2)}%'

            print(info)

            percentage = (epoch + 1) / epochs

            progress_bar.progress(percentage, info)

            train_result.append({
                'epoch': epoch + 1,
                'loss': running_loss / len(train_loader),
                'accuracy': accuracy
            })

        print('Finished Training')

        progress_bar.empty()

        torch.save(model.state_dict(), f'models/{model_name}.pth')

        return train_result
