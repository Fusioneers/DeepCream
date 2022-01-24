from keras.models import Model
from unet_model import unet_model
import tensorflow as tf


model = unet_model(192, 256, 3)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
