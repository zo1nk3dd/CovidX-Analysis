# Refactor all the images to a nice, default size
# This can be 144x144? They start at 299x299

from PIL import Image
import os

data_dir = 'D:\Datasets\COVIDX\Data'
save_dir = 'D:\Datasets\COVIDX\ResizedData'

folders = ['COVID', 'Normal', 'Viral Pneumonia']

IMAGE_SIZE = (128, 128)

for folder in folders:
    for image_fn in os.listdir(os.path.join(data_dir, folder)):
        image = Image.open(os.path.join(data_dir, folder, image_fn))
        res_img = image.resize((IMAGE_SIZE))
        res_img.save(os.path.join(save_dir, folder, image_fn))
