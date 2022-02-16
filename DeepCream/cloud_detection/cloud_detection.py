import logging
import os.path
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import asarray
from pycoral.utils import edgetpu

from DeepCream.constants import ABS_PATH

logger = logging.getLogger('DeepCream.cloud_detection')


class CloudDetection:
    def __init__(self, binary_cloud_threshold: float = 0.85,
                 tpu_support: bool = False):

        """

        Args:

        Optional Args:
            binary_cloud_threshold:
            The threshold (between 0 and 1) which determines if the pixel is
            part of a cloud.

            tpu_support:
            Whether the systems support a tpu (False by default).
        """

        # Set thresholds for cloud detection
        self.binaryCloudThreshold = binary_cloud_threshold

        # Load the machine learning model
        self.HEIGHT = 192
        self.WIDTH = 256
        self.CHANNELS = 3

        if not tpu_support:
            self.interpreter = None
            self.model = tf.keras.models.load_model(os.path.join(ABS_PATH,
                                                 'DeepCream/cloud_detection/models/keras'))
        else:
            self.model = None
            self.interpreter = edgetpu.make_interpreter(
                os.path.join(ABS_PATH, 'DeepCream/cloud_detection/models'
                                       '/tflite/model.tflite'))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def __load_image(self, image: np.ndarray):

        """

        Returns:
            normal:
            The image resized to (self.WIDTH, self.HEIGHT).

            scaled:
            The image resized to (self.WIDTH, self.HEIGHT) and scaled to have
            pixel values between 0 and 1.

        """

        scaled = Image.fromarray(image)
        scaled.thumbnail((self.WIDTH, self.HEIGHT))
        scaled = asarray(scaled)
        scaled = scaled.astype('float32')
        scaled /= 255.0

        return scaled

    def __ai_generate_image_mask(self, image):

        """

        Args:
            image:
            The cloud image in the dimensions
            (self.WIDTH, self.HEIGHT, self.CHANNELS).

        Returns:
            The cloud mask (calculated by the AI) in the dimensions
            (self.WIDTH, self.HEIGHT, 1), with 0 representing a 0% probability
            for a cloud and 255 representing a 100% chance.

        """

        if self.model is not None:
            mask = self.model.predict(np.asarray([image]))
            return mask[0]
        elif self.interpreter is not None:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], image)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            raise ValueError('No AI was configured')

    def evaluate_image(self, image) -> np.ndarray:

        """

        Returns:
            A black and white mask with black representing no clouds and white
            representing clouds.

        """

        # Load the image
        scaled = self.__load_image(image)

        # Check if the image actually loaded
        if scaled is None:
            logger.error('Image was not loaded properly')
            raise ValueError('Image was not loaded properly')

        # Compute the mask
        mask = self.__ai_generate_image_mask(scaled)

        # Make the result binary
        _, mask = cv2.threshold(mask, self.binaryCloudThreshold, 1, cv2.THRESH_BINARY)

        # plt.figure(figsize=(12, 8))
        # plt.subplot(121)
        # plt.title('orig')
        # plt.imshow(scaled)
        # plt.subplot(122)
        # plt.title('mask')
        # plt.imshow(mask)
        # plt.show()

        # Apply the mask to filter out everything but the clouds
        # multi_color_output = cv2.bitwise_and(normal, normal, mask=mask)

        return mask
