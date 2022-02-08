import logging
import os

import cv2 as cv
import numpy as np

from DeepCream.cloud_detection.cloud_filter import CloudFilter
from DeepCream.constants import ABS_PATH


class DeepCream:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def start(self):
        cf = CloudFilter(os.path.join(ABS_PATH, self.input_directory),
                         tpu_support=False)
        mask = cf.evaluate_image('photo_00150_51846468570_o.jpg')
        cv.imwrite(os.path.join(ABS_PATH, self.output_directory, 'test.TIF'),
                   mask)
        return '200 Successful'

    def __get_img(self) -> np.ndarray:
        pass

    def __get_mask(self, orig: np.ndarray) -> np.ndarray:
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

        mask, _ = cloud_filter.evaluate_image(orig)
        logging.debug('Evaluated orig with CloudFilter')

        out = cv.resize(mask, (orig.shape[1], orig.shape[0]))
        logging.debug('Resized mask')

        return out
