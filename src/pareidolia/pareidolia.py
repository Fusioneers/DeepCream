import logging
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from numpy import asarray
import pandas as pd
from typing import List

logger = logging.getLogger('DeepCream.pareidolia')


class Pareidolia:
    def __init__(self):
        """This class is responsible for simulating the pareidolia effect on
        the cloud images.
        """

        # Specify the image parameters
        self.HEIGHT = 224
        self.WIDTH = 224

        # Load the machine learning model
        self.model = tf.keras.models.load_model(
            'src/pareidolia/models/keras/keras_model.h5')

        self.labels = self.__load_labels(
            'src/pareidolia/models/keras/labels.txt')

    @staticmethod
    def __load_labels(labels_file):
        labels = {}
        with open(labels_file, "r") as label:
            text = label.read()
            lines = text.split("\n")
            for line in lines[0:-1]:
                hold = line.split(" ", 1)
                labels[hold[0]] = hold[1]
        return labels

    def __load_image(self, image: np.ndarray) -> np.ndarray:
        """This function is responsible for preparing the image to
        be used as input for the AI.

        Args:
            image:
            The image in RGB format as numpy array.

        Returns:
            normal:
            The image in RGB format as numpy array resized to self.WIDTH and
            self.HEIGHT.

        """

        # Converts the image to a PIL image
        normal = Image.fromarray(image)
        # Resizes the image
        normal = ImageOps.fit(normal, (self.WIDTH, self.HEIGHT),
                              Image.ANTIALIAS)
        # Converts it back to an array
        normal = asarray(normal)
        normal = normal.astype('float32')

        return normal

    def __ai_generate_pareidolia_idea(self, image: np.ndarray):
        """This function generates an array with all the probabilities for each
        label.

        Args:
            image:
            The image in RGB format as numpy array with the dimensions
            self.WIDTH, self.HEIGHT.

        Returns:
            An array with the same length and order as the labels.txt in the
            models/keras and models/tflite folders. Each value in the array
            corresponds to the probability of the cloud image having that
            label. The sum of the probabilities should always give 1.

        """

        if self.model is not None:
            idea = self.model.predict(np.asarray([image]))
            return idea[0]
        else:
            raise ValueError('No AI was configured')

    def __evaluate_image(self, image: np.ndarray) -> str:
        """This function combines the functions above to compute the results
        for an image only containing 1 cloud.

        Args:
            image:
            The image in RGB format as numpy array with any dimensions.

        Returns:
            An array with the same length and order as the labels.txt in the
            models/keras and models/tflite folders. Each value in the array
            corresponds to the probability of the cloud image having that
            label.

        """

        # Load the image
        normal = self.__load_image(image)

        # Check if the image was loaded and formatted correctly
        if normal is None:
            logger.error('Image was not loaded properly')
            raise ValueError('Image was not loaded properly')
        elif normal.shape != (self.HEIGHT, self.WIDTH, 3):
            logger.error('Image has wrong dimensions')
            raise ValueError('Image has wrong dimensions')

        # Compute the probabilities
        pred = self.__ai_generate_pareidolia_idea(normal)

        return pred

    def evaluate_clouds(self, clouds: List[np.ndarray]) -> pd.DataFrame:
        """This is the only public function of the class and its job is to
        call the functions above to compute the probabilities for a list of
        clouds.

        Args:
            clouds:
            An array of image in RGB format as numpy array with any dimensions,
             each only containing 1 cloud.

        Returns:
            A pd.DataFrame with the results to be saved as csv.

        """

        # Check if there is an input
        if not clouds:
            raise ValueError('There are clouds given')

        # Computes the probabilities
        probabilities = []
        for cloud in clouds:
            probabilities.append(self.__evaluate_image(cloud))

        # Writes the probabilities to a data frame in which the labels are the
        # clouds and the probabilities are the values of the individual clouds
        # (rows).
        pareidolia = pd.DataFrame(columns=[self.labels[str(i)]
                                           for i in range(len(self.labels))],
                                  data=probabilities)

        return pareidolia
