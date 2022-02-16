import logging
import os.path
import cv2
import numpy as np
from PIL import Image
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

    def __load_image(self, image: np.ndarray):
        normal = Image.fromarray(image)
        normal.thumbnail((self.WIDTH, self.HEIGHT))
        normal = asarray(normal)
        normal = normal.astype('float32')

        return normal

    def __ai_generate_pareidolia_idea(self, image):
        if self.model is not None:
            idea = self.model.predict(np.asarray([image]))
            return idea[0]
        elif self.interpreter is not None:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], image)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            raise ValueError('No AI was configured')

    def evaluate_image(self, image) -> str:

        normal = self.__load_image(image)

        # Check if the image actually loaded
        if normal is None:
            logger.error('Image was not loaded properly')
            raise ValueError('Image was not loaded properly')

        # Compute the prediction
        pred = self.__ai_generate_pareidolia_idea(image)

        label = pb.labels[str(np.argmax(pred))]

        return label


if __name__ == '__main__':
    pb = Pareidolia(tpu_support=False)

    img = cv2.imread(os.path.join(ABS_PATH, 'data/mug.jpg'))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    res = pb.evaluate_image(img)

    print(res)
