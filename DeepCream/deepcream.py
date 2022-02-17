import logging
import os
import random
import time

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

    def run(self, allowed_execution_time: int):
        print(allowed_execution_time)

        start_time = time.time()

        while int(time.time() - start_time) < allowed_execution_time:
            image = self.__take_photo()

            if self.__review_photo(image):
                identifier = int(self.__save_img(image))
                mask = self.__generate_mask(image)
                self.__save_mask(mask, identifier)
            else:
                plt.figure(figsize=(12, 8))
                plt.subplot(111)
                plt.title('orig')
                plt.imshow(image)
                plt.show()

    def __take_photo(self) -> np.ndarray:
        logger.info('Take photo')
        # Returns a random (RGB) image (placeholder until real camera)
        random_file_name = random.choice(os.listdir(self.directory))
        return cv2.cvtColor(cv2.imread(os.path.join(self.directory, random_file_name)), cv2.COLOR_BGR2RGB)

    def __review_photo(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # TODO find ideal values
        if cv2.mean(gray)[0] < 20 or cv2.mean(gray)[0] > 150:
            return False

        # TODO take the average of the four corners (bigger than 1px)
        if gray[0][0] > 20:
            return False

        return True

    def __save_img(self, image: np.ndarray) -> str:
        return self.database.save_orig(image)

    def __generate_mask(self, image):
        logger.info('Generate mask')
        return self.cloud_detection.evaluate_image(image)

    def __save_mask(self, mask: np.ndarray, identifier: int) -> int:
        logger.info('Save mask for image ' + str(identifier))
        return self.database.save_mask(mask, str(identifier))

    def __get_analysis(self, image: np.ndarray,
                       mask: np.ndarray) -> Analysis:
        pass

    def __interpretation(self, analysis: Analysis):
        pass

    def __save_results(self,
                       analysis_: Analysis,
                       interpretation):
        pass

    def __load_img(self, identifier: int) -> np.ndarray:
        return self.database.load_orig_by_id(str(identifier))
