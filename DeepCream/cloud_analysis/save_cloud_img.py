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

columns = ['center x',
           'center y',
           'contour perimeter',
           'contour area',
           'hull perimeter',
           'hull area',
           'roundness',
           'convexity',
           'solidity',
           'rectangularity',
           'elongation',
           'mean r',
           'mean g',
           'mean b',
           'std r',
           'std g',
           'std b',
           'std',
           'transparency',
           'sharp edges']

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

        analysis = Analysis(img, mask, 10, 1)
        df = pd.DataFrame(columns=columns)

        for j, cloud in enumerate(analysis.clouds):
            std = cloud.std()
            mean = cloud.mean()
            try:
                diff_edges = cloud.diff_edges(50,
                                              200)
            except ValueError as err:
                logger.warning(err)
                continue

            df.loc[j, ['center x']] = cloud.center[0]
            df.loc[j, ['center y']] = cloud.center[1]
            df.loc[
                j, ['contour perimeter']] = cloud.contour_perimeter
            df.loc[j, ['contour area']] = cloud.contour_area
            df.loc[j, ['hull area']] = cloud.hull_area
            df.loc[j, ['hull perimeter']] = cloud.hull_perimeter
            df.loc[j, ['convexity']] = cloud.convexity()
            df.loc[j, ['roundness']] = cloud.roundness()
            df.loc[j, ['solidity']] = cloud.solidity()
            df.loc[j, ['rectangularity']] = cloud.rectangularity()
            df.loc[j, ['elongation']] = cloud.elongation()

            df.loc[j, ['mean r']] = mean[0]
            df.loc[j, ['mean g']] = mean[1]
            df.loc[j, ['mean b']] = mean[2]
            df.loc[j, ['std r']] = std[0]
            df.loc[j, ['std g']] = std[1]
            df.loc[j, ['std b']] = std[2]
            df.loc[j, ['std']] = sum(std) / 3
            df.loc[j, ['transparency']] = cloud.transparency()
            df.loc[j, ['sharp edges']] = np.mean(diff_edges)

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
