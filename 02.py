import os
from PIL import Image


def split_image_into_parts(input_dir, output_dir):
    """
    Splits each image in the input directory into 9 parts and saves the parts
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
            step_w, step_h = width // 4, height // 4

            # Generate and save 9 cropped images
            for i in range(4):
                for j in range(4):
                    left = i * step_w
                    upper = j * step_h
                    right = (i + 1) * step_w
                    lower = (j + 1) * step_h
                    cropped_img = img.crop((left, upper, right, lower))

                    # Construct a filename for the cropped image
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}_{i}_{j}{os.path.splitext(image_name)[1]}"
                    cropped_img.save(os.path.join(output_class_dir, cropped_image_name))


# Example usage
input_dir = 'D:\Master Project\model\model-1\Corallo-vs-Myxo\\train'
output_dir = 'D:\Master Project\model\model-1\Corallo-vs-Myxo\\train2'
split_image_into_parts(input_dir, output_dir)
