import logging
import os
import traceback

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from DeepCream.classification.classification import Classification
from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import (DEBUG_MODE,
                                 ABS_PATH,
                                 DEFAULT_BORDER_WIDTH,
                                 get_time,
                                 )
from DeepCream.database import DataBase

logger = logging.getLogger('DeepCream.save_cloud_img')
input_dir = os.path.normpath(os.path.join(ABS_PATH, 'data/input'))
output_dir = os.path.normpath(
    os.path.join(ABS_PATH, f'data/database {get_time()}'))
num_img = len(os.listdir(input_dir))

cloud_detection = CloudDetection()
database = DataBase(output_dir)
classification = Classification()

for i, path in tqdm(enumerate(os.scandir(input_dir)), total=num_img):
    try:
        logger.info(f'Reading image {path.name}')
        img = cv.cvtColor(cv.imread(os.path.normpath(path.path)),
                          cv.COLOR_BGR2RGB)
        if not img.size:
            logger.error('Orig was loaded empty')

        identifier = database.save_orig(img)

        mask = cloud_detection.evaluate_image(img).astype('uint8') * 255

        database.save_mask(mask, identifier)

        analysis = Analysis(img, mask, 10, 1).evaluate()

        database.save_analysis(df, identifier)

        classification_ = classification.get_classification(df)
        database.save_classification(classification_, identifier)

        if DEBUG_MODE:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
            axes[0].imshow(analysis.clouds[0].img)
            axes[1].imshow(img)
            axes[2].imshow(mask)
            fig.suptitle(f'Type of largest cloud: '
                         f'{classification_.loc[0].idxmax()}')
            plt.show()

    except (ValueError,
            TypeError,
            IndexError,
            FileExistsError,
            FileNotFoundError,
            ArithmeticError,
            NameError,
            LookupError,
            AssertionError,
            RecursionError,
            ):
        logger.error(traceback.format_exc())
