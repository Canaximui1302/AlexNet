import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pathlib 
import tensorflow as tf
import numpy as np
import datetime


data_dir = pathlib.Path("./flower_photos")
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count: " + str(image_count))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"] )
print("Class names: ") 
print(CLASS_NAMES)

output_class_units = len(CLASS_NAMES)
print("Output units: " + str(output_class_units))

#Alexnet model


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(227, 227, 3)),
        #conv 1
    tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
        # 2nd conv
    tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
        # 3rd conv
    tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
        # 4th conv
    tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
        # 5th Conv
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1,1), activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
    # To Flatten layer
    tf.keras.layers.Flatten(),
    # To FC layer 1
    tf.keras.layers.Dense(4096, activation='relu'),
        # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
    #To FC layer 2
    tf.keras.layers.Dense(4096, activation='relu'),
        # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(output_class_units, activation='softmax')




])




BATCH_SIZE = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227
Steps_per_epoch = int(np.floor(image_count/BATCH_SIZE))

img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = img_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))


model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ['accuracy'])

model.summary()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get("accuracy")==1.00 and logs.get("loss")<0.03):
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training =True
callbacks = myCallback()
log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Training the Model
history = model.fit(
      train_data_gen,
      steps_per_epoch=Steps_per_epoch,
      epochs=20)

# Saving the model
model.save('AlexNet_saved_model.keras')








