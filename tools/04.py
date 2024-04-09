import os
import matplotlib.pyplot as plt

# Path to your dataset folder
dataset_folder = "D:\Master Project\Data\Image\Iran Data"
dataset_folder2 = "D:\Master Project\Data\Image\Germany Data"

# Function to count images in each subfolder
def count_images_per_class(dataset_folder):
    class_count = {}
    for class_folder in os.listdir(dataset_folder):
        if os.path.isdir(os.path.join(dataset_folder, class_folder)):
            num_images = len(os.listdir(os.path.join(dataset_folder, class_folder)))
            class_count[class_folder] = num_images
    return class_count

# Get the count of images per class
class_count = count_images_per_class(dataset_folder)

# Plotting the counts
plt.figure(figsize=(10, 6))
bars = plt.bar(class_count.keys(), class_count.values(), color='lightsalmon')
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add text annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '%d' % int(height),
             ha='center', va='bottom')

plt.show()