import logging
import os.path
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from numpy import asarray
from pycoral.utils import edgetpu

from DeepCream.constants import ABS_PATH

logger = logging.getLogger('DeepCream.pareidolia')


class Pareidolia:
    def __init__(self, tpu_support: bool = False):

        # Load the machine learning model
        self.HEIGHT = 224
        self.WIDTH = 224

        if not tpu_support:
            self.interpreter = None
            self.model = tf.keras.models.load_model(os.path.join(ABS_PATH,
                                                                 'DeepCream/pareidolia/models/keras/keras_model.h5'))

            self.labels = self.__load_labels(os.path.join(ABS_PATH, 'DeepCream/pareidolia/models/keras/labels.txt'))
        else:
            self.model = None
            self.interpreter = edgetpu.make_interpreter(
                os.path.join(ABS_PATH, 'DeepCream/pareidolia/models/tflite/model.tflite'))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.labels = self.__load_labels(os.path.join(ABS_PATH, 'DeepCream/pareidolia/models/tflite/labels.txt'))

    def __load_labels(self, labels_file):
        labels = {}
        with open(labels_file, "r") as label:
            text = label.read()
            lines = text.split("\n")
            for line in lines[0:-1]:
                hold = line.split(" ", 1)
                labels[hold[0]] = hold[1]
        return labels

    def __load_image(self, image: np.ndarray) -> np.ndarray:
        normal = Image.fromarray(image)
        normal = ImageOps.fit(normal, (self.WIDTH, self.HEIGHT), Image.ANTIALIAS)
        # normal.thumbnail((self.WIDTH, self.HEIGHT))
        normal = asarray(normal)
        normal = normal.astype('float32')

        return normal

    def __ai_generate_pareidolia_idea(self, image: np.ndarray):
        if self.model is not None:
            idea = self.model.predict(np.asarray([image]))
            return idea[0]
        elif self.interpreter is not None:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], np.asarray([image]))
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            raise ValueError('No AI was configured')

    def evaluate_image(self, image: np.ndarray) -> str:

        normal = self.__load_image(image)

        # Check if the image actually loaded
        if normal is None:
            logger.error('Image was not loaded properly')
            raise ValueError('Image was not loaded properly')
        elif normal.shape != (self.HEIGHT, self.WIDTH, 3):
            logger.error('Image has wrong dimensions')
            raise ValueError('Image has wrong dimensions')

        # Compute the prediction
        pred = self.__ai_generate_pareidolia_idea(normal)

        label = pb.labels[str(np.argmax(pred))]

        return label


if __name__ == '__main__':
    pb = Pareidolia(tpu_support=False)

    img = cv2.imread(os.path.join(ABS_PATH, 'data/input/camel.jpg'))

    res = pb.evaluate_image(img)

    print(res)
