from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = 'dataset'
labels = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry',
          'Soybean', 'Squash', 'Strawberry', 'Tomato']

datagen = ImageDataGenerator(
    channel_shift_range=100
)

imagegen = datagen.flow_from_directory('dataset/training', batch_size=3)


num_row = 4
num_col = 3
fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_row, 6*num_col))
for i in range(num_row*num_col):
  if i < 12:
    class_imgs = next(os.walk('{}/training/{}/'.format(dataset_dir, labels[i])))[2]
    class_img = class_imgs[0]
    img = next(imagegen)[0][0]

    ax = axes[i//num_col, i % num_col]
    ax.imshow(np.uint8(img))
    ax.set_title('{}'.format(labels[i]))
plt.tight_layout()
plt.show()
