import datetime
import os
import torch
import shutil
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import ImageFolder


def save_dataset_to_path(dataset, path):
    """
    Saves a dataset to a specified path. Assumes dataset is iterable of (image, label) tuples.

    Args:
    dataset: An iterable of tuples (image, label), where `image` can be a PIL Image or a file path.
    path: The root directory where the dataset will be saved.

    Returns:
    None
    """
    print('dataset length: ', len(dataset))
    t = datetime.datetime.now()

    # Create the root directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    class_names = dataset.classes

    for idx, (image, label) in enumerate(dataset):
        # Define the directory for the current label, if it doesn't exist, create it
        label_dir = os.path.join(path, str(class_names[label]))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Save the tensor as an image
        image_file_path = os.path.join(label_dir, f'image_{idx}.png')

        # Check the type of image and handle accordingly
        if isinstance(image, str):  # image is a filepath
            shutil.copy(image, image_file_path)
        elif isinstance(image, Image.Image):  # image is a PIL Image
            image.save(image_file_path)
        elif isinstance(image, torch.Tensor):  # image is a PyTorch Tensor
            save_image(image, image_file_path)

    delta_time = datetime.datetime.now() - t
    print('time: ', delta_time)



# Example usage:
# Assuming `dataset` is a list of tuples [(image_path, label), ...]
# save_dataset_to_path(dataset, '/path/to/save/dataset')
# data_path = 'D:\Master Project\model\model-1\Corallo-vs-Myxo-224-split\\val'
# dataset = ImageFolder(root=data_path)
# save_dataset_to_path(dataset, path='D:\Master Project\model\model-1\Corallo-vs-Myxo-224-split2\\val')
