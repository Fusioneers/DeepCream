import logging
import os

import cv2 as cv
import numpy as np
from PIL import Image

from DeepCream.cloud_detection.cloud_filter import CloudFilter
from DeepCream.constants import ABS_PATH


class DeepCream:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def start(self):
        pass

    def __get_img(self) -> np.ndarray:
        pass

    def __get_mask(self, path: str) -> np.ndarray:
        """Gets the cloud mask from CloudFilter.

        Returns:
            A numpy array which corresponds to an image which has the same
            height and width as orig, with a 255 for a pixel which is
            estimated to be a cloud and a 0 otherwise. It has the same shape
            as orig, but only a single channel i.e. shape (height, width).
        """
        # TODO convert directory to image
        cloud_filter = CloudFilter()
        logging.debug('Initialised CloudFilter')

        mask, _ = cloud_filter.evaluate_image(Image.open(path))
        logging.debug('Evaluated image with CloudFilter')

        out = cv.resize(mask, (mask.shape[1], mask.shape[0]))

        return out
