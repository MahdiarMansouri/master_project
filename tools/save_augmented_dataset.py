from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, autoaugment
from torch.utils.data import Dataset, DataLoader
import os
import torch
from save_dataset import save_dataset_to_path


class CustomAugmentedDataset(Dataset):
    def __init__(self, root_dir, num_samples_per_class, transform=None, num_magnitude_bins=30):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_samples_per_class (int): Desired number of samples per class after augmentation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = ImageFolder(root=root_dir)
        self.classes = self.dataset.classes
        self.num_samples_per_class = num_samples_per_class
        self.transform = transform
        self.augment_transform = transforms.Compose([
            autoaugment.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins),
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        self.class_samples = self._balance_classes()

    def _balance_classes(self):
        from collections import defaultdict
        class_indices = defaultdict(list)

        for idx, (_, class_id) in enumerate(self.dataset.samples):
            class_indices[class_id].append(idx)

        # Reduce or oversample class indices to match num_samples_per_class
        balanced_indices = []
        for indices in class_indices.values():
            if len(indices) >= self.num_samples_per_class:
                balanced_indices.extend(indices[:self.num_samples_per_class])
            else:
                # Oversample if there are fewer samples than desired
                oversampled_indices = indices * (self.num_samples_per_class // len(indices)) + indices[
                                                                                               :self.num_samples_per_class % len(
                                                                                                   indices)]
                balanced_indices.extend(oversampled_indices)

        return balanced_indices

    def __len__(self):
        return len(self.class_samples)

    def __getitem__(self, idx):
        img, label = self.dataset[self.class_samples[idx]]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.augment_transform(img)
        return img, label

    def classes(self):
        return self.classes


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Define Parameters
data_path = 'D:\Master Project\model\model-1\myxo-vs-nonmyxo-V2-9p-filtered-3classes'
num_magnitude_bins = 100
train_num_samples_per_class = 10000
val_num_samples_per_class = 2000

# Define any additional transformations
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flipping the image horizontally
    transforms.RandomRotation(degrees=20),  # Rotate by -20 to +20 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
    transforms.ColorJitter(brightness=0.3),  # Adjusting Brightness
    transforms.ColorJitter(contrast=0.3),  # Adjusting Contrast
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # AddGaussianNoise(mean=-0.1, std=0.3),
])

# Creating datasets
datasets = {
    x: CustomAugmentedDataset(root_dir=os.path.join(data_path, x), num_magnitude_bins=num_magnitude_bins,
                              transform=aug_transform,
                              num_samples_per_class=train_num_samples_per_class if x == 'train' else val_num_samples_per_class)
    for x in ['train', 'val']
}

print('Datasets created.')

# Show Classes
class_names = datasets['train'].classes
print(f'there are {len(class_names)} classes, and class names are {class_names}')
print('-' * 50)

# Show datasets length
print('train dataset: ', len(datasets['train']))
print('val dataset: ', len(datasets['val']))

save_dataset_to_path(datasets['train'], 'D:\Master Project\model\model-1\myxo-vs-nonmyxo-V2-9p-filtered-3classes-augmented\\train')
save_dataset_to_path(datasets['val'], 'D:\Master Project\model\model-1\myxo-vs-nonmyxo-V2-9p-filtered-3classes-augmented\\val')

