import torchvision.transforms
from torchvision.transforms import TrivialAugmentWide, InterpolationMode
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import torch
import matplotlib.pyplot as plt

img = Image.open('swim.jpg')
# convertor = torch.Tensor()
tensor_img = pil_to_tensor(img)
# print(tensor_img)

trivial = TrivialAugmentWide(num_magnitude_bins=100, interpolation=InterpolationMode.BILINEAR)

trivial_img = trivial(tensor_img)
# print(trivial_img)

plt.imshow(trivial_img.permute(1, 2, 0))
plt.show()