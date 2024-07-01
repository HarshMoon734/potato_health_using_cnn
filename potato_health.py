import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

IMAGE_SIZE = 256

_dirname = r'C:\Users\Harsh\Downloads\potato_\PlantVillage\real'

def see_image(training, index):
    files = os.listdir(training)
    img = mpimg.imread(f"{training}\{files[index]}")
    plt.imshow(img)
    plt.show()

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    _dirname,
    shuffle = True,
    image_size = (256,256),
    batch_size = 32
)

class_names_length = train_dataset.class_names

train_dataset = train_dataset.take(int(len(train_dataset)*80/100))
validation_dataset = train_dataset.take(int(len(train_dataset)*10/100))
test_dataset = train_dataset.take(int(len(train_dataset)*10/100))

model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_names_length), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=50, validation_data=validation_dataset, verbose = 1)

print(train_dataset.class_names)
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

# print(model.evaluate(test_dataset)) # if you want to evaluate the model on the bases of test_dataset

