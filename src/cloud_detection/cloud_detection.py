import logging
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from numpy import asarray

logger = logging.getLogger('DeepCream.cloud_detection')


class CloudDetection:
    def __init__(self, binary_cloud_threshold: float = 0.85):
        """This class is responsible for filtering out the clouds of any given
        image.

        Args:
            binary_cloud_threshold:
            The threshold (between 0 and 1) which determines if the pixel is
            part of a cloud. This value is used in evaluate_image after and
            applied to the AI generated mask. (0.85 by default)
        """

        # Set thresholds for cloud detection
        self.binaryCloudThreshold = binary_cloud_threshold

        # Specify the image parameters
        self.HEIGHT = 192
        self.WIDTH = 256
        self.CHANNELS = 3

        # Load the machine learning model
        self.model = tf.keras.models.load_model('src/cloud_detection/models/keras')

    def __load_image(self, image: np.ndarray) -> np.ndarray:
        """This function is responsible for preparing the image to be used as
        input for the AI.

        Args:
            image:
            The image in RGB format as numpy array.

        Returns:
            scaled:
            The image in RGB format as numpy array resized to self.WIDTH and
            self.HEIGHT and scaled to have pixel values between 0 and 1.

        """

        # Converts the image to a PIL image
        scaled = Image.fromarray(image)
        # Resizes the image
        scaled = ImageOps.fit(scaled, (self.WIDTH, self.HEIGHT),
                              Image.ANTIALIAS)

        # Converts it back to an array
        scaled = asarray(scaled)
        scaled = scaled.astype('float32')
        # Scales the pixel values to lie between 0 and 1
        scaled /= 255.0

        return scaled

    def __ai_generate_image_mask(self, image: np.ndarray) -> np.ndarray:
        """This function generates an image mask based on the input image.

        Args:
            image:
            The image in RGB format as numpy array with the dimensions
            self.WIDTH, self.HEIGHT.

        Returns:
            The cloud mask (calculated by the AI) as black and white image in
            the dimensions self.WIDTH, self.HEIGHT, with 1 representing a 100%
            chance for a pixels to be part of a cloud and 0 representing a 0%
            chance.

        """

        if self.model is not None:
            mask = self.model.predict(np.asarray([image]))
            return mask[0]
        else:
            raise ValueError('No AI was configured')

    def evaluate_image(self, image: np.ndarray) -> np.ndarray:
        """This is the only public function of the class and its job is to
        call the functions above to then return the mask.

        Args:
            image:
            The image in RGB format as numpy array with any dimensions.

        Returns:
            A black and white mask as numpy array with white representing
            cloud pixels and black representing all others.

        """

        # Load the image
        scaled = self.__load_image(image)

        # Check if the image was loaded and formatted correctly
        if scaled is None:
            logger.error('Image was not loaded properly')
            raise ValueError('Image was not loaded properly')
        elif scaled.shape != (self.HEIGHT, self.WIDTH, 3):
            logger.error('Image has wrong dimensions')
            raise ValueError('Image has wrong dimensions')

        # Compute the mask
        mask = self.__ai_generate_image_mask(scaled)

        # Make the result binary (using the threshold from the beginning)
        _, mask = cv2.threshold(mask, self.binaryCloudThreshold, 1,
                                cv2.THRESH_BINARY)

        return mask
