from DeepCream.constants import TIME_FORMAT, LOGGING_FORMAT, LOG_PATH, ABS_PATH
import os
import logging
import cv2 as cv
from DeepCream.cloud_analysis.analysis import Analysis

with open(LOG_PATH, 'w') as log:
    logging.basicConfig(
        filename=LOG_PATH,
        format=LOGGING_FORMAT, level=logging.DEBUG)

# path = os.path.normpath(
#     os.path.join(REP_PATH, 'sample_data/Data/zz_astropi_1_photo_364.jpg'))
#
# img = cv.imread(path)
# analysis = Analysis(img, 2000, 20, 100)
