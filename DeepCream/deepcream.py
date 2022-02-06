import os

import cv2

from DeepCream.cloud_detection.cloud_filter import CloudFilter
from DeepCream.constants import ABS_PATH


class DeepCream:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def start(self):
        cf = CloudFilter(os.join(ABS_PATH, self.input_directory),
                         tpu_support=False)
        mask = cf.evaluate_image('photo_00150_51846468570_o.jpg')
        cv2.imwrite(os.join(ABS_PATH, self.output_directory, 'test.TIF'),
                    mask)
        return '200 Successful'
