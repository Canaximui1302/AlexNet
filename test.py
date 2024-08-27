import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pathlib 
import tensorflow as tf
import numpy as np
import datetime



CLASS_NAMES = np.array(["daisy", "dandelion", "roses", "sunflowers", "tulips"] )


BATCH_SIZE = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227

model = tf.keras.models.load_model("AlexNet_saved_model.keras")

while(True):
    inp = input("Image file: ")
    if inp == "q":
        break

    inp = "predict/" + inp
    image = tf.keras.preprocessing.image.load_img(inp, target_size=(227, 227))

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    input_arr = input_arr.astype('float32') / 255.
    #im = image.fromarray((input_arr * 255).astype(np.uint8))




    
    predictions = model.predict(input_arr)
    print("\nResult: ", CLASS_NAMES[np.argmax(predictions)])