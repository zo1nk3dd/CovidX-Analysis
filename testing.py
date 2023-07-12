from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype
from torch import nn
from model import Classifier
import torch
image = read_image('D:\Datasets\COVIDX\Data\COVID\COVID-1.png')
image = ConvertImageDtype(torch.float32)(image)
image = torch.unsqueeze(image, 0)
model = Classifier()

model.training_step((image, 1), 0)