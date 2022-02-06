import cv2
from DeepCream.cloud_detection.cloud_filter import CloudFilter
import logging
from DeepCream.constants import (LOGGING_FORMAT, ABS_PATH, LOGGING_LEVEL, LOG_PATH)


class DeepCream:
    def __init__(self, input_directory, output_directory):
        with open(LOG_PATH, 'w') as log:
            print('Opened log file: ' + str(LOG_PATH))
            logging.basicConfig(
                filename=LOG_PATH,
                format=LOGGING_FORMAT, level=LOGGING_LEVEL)
            print('Successfully configured logging')
            logging.info('Started DeepCream/__init__.py')

        self.input_directory = input_directory
        self.output_directory = output_directory

    def start(self):
        cf = CloudFilter(ABS_PATH + "/" + self.input_directory, tpu_support=False)
        mask = cf.evaluate_image("photo_00150_51846468570_o.jpg")
        cv2.imwrite(ABS_PATH + "/" + self.output_directory + "/" + "test.TIF", mask)
        return "200 Successful"
