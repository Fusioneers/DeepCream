import logging
import os

import cv2 as cv
import numpy as np
from PIL import Image
import threading as th

from cloud_detection.cloud_filter import CloudFilter
from cloud_analysis.analysis import Analysis
from database import DataBase
from constants import ABS_PATH


class DeepCream:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def start(self):
        pass

    def __get_img(self) -> np.ndarray:
        pass

    def __save_img(self, img: np.ndarray):
        pass

    def __load_img(self, directory):
        pass

    def __get_mask(self, img: np.ndarray) -> np.ndarray:
        """Gets the cloud mask from CloudFilter.

        Returns:
            A numpy array which corresponds to an image which has the same
            height and width as orig, with a 255 for a pixel which is
            estimated to be a cloud and a 0 otherwise. It has the same shape
            as orig, but only a single channel i.e. shape (height, width).
        """
        
        cloud_filter = CloudFilter()
        logging.debug('Initialised CloudFilter')

        mask, _ = cloud_filter.evaluate_image(Image.fromarray(img, mode="RGB"))
        logging.debug('Evaluated image with CloudFilter')

        out = cv.resize(mask, (mask.shape[1], mask.shape[0]))

        return out

    def __get_analysis(self,
                       img: np.ndarray,
                       mask: np.ndarray) -> Analysis:
        pass

    def __interpretation(self, analysis: Analysis):
        pass

    def __save_results(self,
                       analysis_: Analysis,
                       interpretation: Interpretation):
        pass
