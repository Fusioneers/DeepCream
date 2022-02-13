import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import ABS_PATH
from DeepCream.database import DataBase

logger = logging.getLogger('DeepCream.deepcream')


class DeepCream:
    def __init__(self, directory: str, tpu_support: bool):
        self.directory = directory

        self.cloud_detection = CloudDetection(tpu_support=tpu_support)
        self.database = DataBase(os.path.join(ABS_PATH, 'database'))

    def start(self):
        for i in range(1, 11):
            image = self.__load_img('photo_' + str(i) + '.jpg')
            identifier = self.database.save_orig(image)
            mask = self.__get_mask(image)
            self.database.save_mask(mask, identifier)

    def __get_img(self) -> np.ndarray:
        pass

    def __save_img(self, img: np.ndarray):
        pass

    def __load_img(self, image_name: str) -> np.ndarray:
        # Returns an RGB (!) image
        return cv2.cvtColor(cv2.imread(os.path.join(self.directory, image_name)
                                       ), cv2.COLOR_BGR2RGB)

    def __get_mask(self, image):
        return self.cloud_detection.evaluate_image(image)

    def __get_analysis(self, image: np.ndarray,
                       mask: np.ndarray) -> Analysis:
        pass

    def __interpretation(self, analysis: Analysis):
        pass

    def __save_results(self,
                       analysis_: Analysis,
                       interpretation):
        pass
