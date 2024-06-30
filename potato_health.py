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

train_dir = r'C:\Users\Harsh\Downloads\potato_\PlantVillage\real\Potato___Early_blight'
val_dir = r'C:\Users\Harsh\Downloads\potato_\PlantVillage\real\Potato___healthy'
test_dir = r'C:\Users\Harsh\Downloads\potato_\PlantVillage\real\Potato___Late_blight'

def see_image(training, index):
    files = os.listdir(training)
    # print(files[0])
    img = mpimg.imread(f"{training}\{files[index]}")
    plt.imshow(img)
    plt.show()

files = os.listdir(test_dir)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Harsh\Downloads\potato_\PlantVillage\real",
    shuffle = True,
    image_size = (256,256),
    batch_size = 32
)

train_dataset = dataset.take(int(len(dataset)*80/100))
test_dataset = dataset.take(int(len(dataset)*10/100))
validation_dataset = dataset.take(int(len(dataset)*10/100))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

# print(model.evaluate(test_dataset)) # if you want to evaluate the model on the bases of test_dataset

