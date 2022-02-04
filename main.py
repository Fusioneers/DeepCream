from DeepCream.constants import time_format, logging_format, log_path, rep_path
import os
import logging
import cv2 as cv
from DeepCream.cloud_analysis.analysis import Analysis

with open(log_path, 'w') as log:
    logging.basicConfig(
        filename=log_path,
        format=logging_format, level=logging.DEBUG,
        datefmt=time_format)

# path = os.path.normpath(
#     os.path.join(rep_path, 'sample_data/Data/zz_astropi_1_photo_364.jpg'))
#
# img = cv.imread(path)
# analysis = Analysis(img, 2000, 20, 100)
