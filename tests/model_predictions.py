import torch
from torchvision import transforms
from PIL import Image

dinov2_vits14_reg_21M = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vitb14_reg_86M = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

model1 = torch.load('../models/model_1.pth')
model2 = torch.load('../models/model_2.pth')
model3 = torch.load('../models/model_3.pth')
model4 = torch.load('../models/model_4.pth')
model5 = torch.load('../models/model_2_3.pth')



# Ensure model is in evaluation mode
model_list = [model1, model2, model3, model4, model5]

# Path to your image
image_path = 'photo_4_2024-04-12_18-53-48.jpg'

# Define the same transforms as used during model training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming you used this size during training
    transforms.ToTensor(),
    # Make sure to include the same normalization as during training
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image = Image.open(image_path).convert('RGB')

# Preprocess the image
image_tensor = transform(image)

# Add batch dimension (model expects batches, we're using batch size of 1 here)
image_tensor = image_tensor.unsqueeze(0)

# Move the input to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)

for idx, model in enumerate(model_list):
    model = model.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert output probabilities to predicted class
    _, preds = torch.max(output, 1)
    predicted_label = preds.item()
    predicted_label = 'Corallococcus' if predicted_label == 0 else 'Myxococcus'

    # Print the predicted label
    print(f'model {idx + 1} | Predicted label --> {predicted_label}')
