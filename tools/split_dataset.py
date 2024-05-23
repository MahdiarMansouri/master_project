from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split


def split_image_into_parts(input_dir, output_dir, split_number):
    """
    Splits each image in the input directory into split_number parts and saves the parts
    in a mirrored directory structure in the output directory.

    Args:
    - input_dir (str): Path to the input directory containing class subfolders with images.
    - output_dir (str): Path to the output directory where cropped images will be saved.
    """
    # Iterate over all the class subdirectories in the input directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip files

        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)  # Create class subdirectory in output

        # Process each image in the class subdirectory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if not os.path.isfile(image_path):
                continue  # Skip subdirectories

            img = Image.open(image_path)
            width, height = img.size
            step_w, step_h = width // split_number, height // split_number

            # Generate and save 9 cropped images
            for i in range(split_number):
                for j in range(split_number):
                    left = i * step_w
                    upper = j * step_h
                    right = (i + 1) * step_w
                    lower = (j + 1) * step_h
                    cropped_img = img.crop((left, upper, right, lower))

                    # Construct a filename for the cropped image
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}_{i}_{j}{os.path.splitext(image_name)[1]}"
                    cropped_img.save(os.path.join(output_class_dir, cropped_image_name))


def split_by_size(input_dir, output_dir, split_size=224):
    """
    Splits each image in the input directory into split_size images (224, 224) and saves the parts
    in a mirrored directory structure in the output directory.

    Args:
    - input_dir (str): Path to the input directory containing class subfolders with images.
    - output_dir (str): Path to the output directory where cropped images will be saved.
    """
    # Iterate over all the class subdirectories in the input directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip files

        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)  # Create class subdirectory in output

        # Process each image in the class subdirectory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if not os.path.isfile(image_path):
                continue  # Skip subdirectories

            img = Image.open(image_path)
            width, height = img.size
            step_w, step_h = width // split_size, height // split_size

            # Generate and save cropped images
            for i in range(step_w):
                for j in range(step_h):
                    left = i * split_size
                    upper = j * split_size
                    right = (i + 1) * split_size
                    lower = (j + 1) * split_size
                    cropped_img = img.crop((left, upper, right, lower))

                    # Construct a filename for the cropped image
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}_{i}_{j}{os.path.splitext(image_name)[1]}"
                    cropped_img.save(os.path.join(output_class_dir, cropped_image_name))


def split_dataset(root_dir, output_dir, validation_ratio):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List of classes
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        images = os.listdir(cls_path)

        train_images, val_images = train_test_split(images, test_size=validation_ratio)

        # Create class directories in train and val folders
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))

        # Copy validation images
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

    print("Dataset split completed!")


# Usage
root_dir = 'D:\\Dataset\segment_dataset-2'
output_dir = 'D:\\Dataset\segment_dataset-2-split'  # Output path for train/val folders
validation_ratio = 0.2
# split_dataset(root_dir, output_dir, validation_ratio)

# Example usage
input_dir = 'D:\Master Project\model\model-1\segment_dataset-2\\val'
output_dir = 'D:\Master Project\model\model-1\segment_dataset-2-split\\val'
# split_by_size(input_dir, output_dir, 224)
split_image_into_parts(input_dir, output_dir, 3)
