import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection


class DeepCream:
    def __init__(self, directory: str):
        self.directory = directory

        self.cloud_detection = CloudDetection(tpu_support=True)

    def start(self):
        image = self.__load_img('error.jpg')
        mask = self.__get_mask(image)

        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.title('image')
        plt.imshow(image)
        plt.subplot(122)
        plt.title('mask')
        plt.imshow(mask)
        plt.show()

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
