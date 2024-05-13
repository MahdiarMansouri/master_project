import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SplitImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split_number=3):
        """
        Args:
            root_dir (string): Directory with all the images and subdirectories as classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load the dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=root_dir)

        # Create the (image, label) pairs, where each dimension of image is split into (split_number) parts.
        for img, label in self.dataset:
            width, height = img.size
            step_w, step_h = width // split_number, height // split_number

            for i in range(split_number):
                for j in range(split_number):
                    left = i * step_w
                    upper = j * step_h
                    right = (i + 1) * step_w
                    lower = (j + 1) * step_h
                    self.data.append(img.crop((left, upper, right, lower)))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

split_dataset = SplitImageDataset(root_dir='D:\Master Project\model\model-1\Corallo-vs-Myxo\\train',
                                  transform=transform, split_number=4)
dataloader = DataLoader(split_dataset, batch_size=32, shuffle=True)

print(split_dataset.__len__())
