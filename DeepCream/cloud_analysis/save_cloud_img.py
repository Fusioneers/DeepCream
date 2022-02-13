import logging
import os

import cv2 as cv
import numpy as np
import pandas as pd

import traceback

from tqdm import tqdm
from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.database import DataBase
from DeepCream.constants import ABS_PATH, DEFAULT_BORDER_WIDTH

logger = logging.getLogger('DeepCream.save_cloud_img')
input_dir = os.path.normpath(os.path.join(ABS_PATH, 'data/input'))
output_dir = os.path.normpath(os.path.join(ABS_PATH, 'data/database'))
num_img = len(os.listdir(input_dir))

cloud_detection = CloudDetection()
database = DataBase(output_dir)

columns = ['center_x',
           'center_y',
           'contour_perimeter',
           'contour_area',
           'hull_perimeter',
           'hull_area',
           'rectangularity',
           'elongation',
           'mean_r',
           'mean_g',
           'mean_b',
           'std_r',
           'std_g',
           'std_b',
           'transparency',
           'mean_diff_edges']

for i, path in tqdm(enumerate(os.scandir(input_dir)), total=num_img):
    try:
        logger.info(f'Reading image {path.name}')
        img = cv.cvtColor(cv.imread(os.path.normpath(path.path)),
                          cv.COLOR_BGR2RGB)
        identifier = database.save_orig(img, is_compressed=True)
        logger.info('Saved orig')

        mask = cloud_detection.evaluate_image(img)
        database.save_mask(mask, identifier)
        logger.info('Saved mask')

        analysis = Analysis(img, mask, 5, 0.5)
        df = pd.DataFrame(columns=columns)

        for j, cloud in enumerate(analysis.clouds):
            std = cloud.std()
            mean = cloud.mean()

            df.loc[j, ['center_x']] = cloud.center[0]
            df.loc[j, ['center_y']] = cloud.center[1]
            df.loc[
                j, ['contour_perimeter']] = cloud.contour_perimeter
            df.loc[j, ['contour_area']] = cloud.contour_area
            df.loc[j, ['hull_area']] = cloud.hull_area
            df.loc[j, ['hull_perimeter']] = cloud.hull_perimeter
            df.loc[j, ['rectangularity']] = cloud.rectangularity()
            df.loc[j, ['elongation']] = cloud.elongation()

            df.loc[j, ['mean_r']] = mean[0]
            df.loc[j, ['mean_g']] = mean[1]
            df.loc[j, ['mean_b']] = mean[2]
            df.loc[j, ['std_r']] = std[0]
            df.loc[j, ['std_g']] = std[1]
            df.loc[j, ['std_b']] = std[2]
            df.loc[j, ['transparency']] = cloud.transparency()
            df.loc[j, ['mean_diff_edges']] = np.mean(
                cloud.diff_edges(DEFAULT_BORDER_WIDTH, DEFAULT_BORDER_WIDTH))

        database.save_analysis(df, identifier)
        logger.info('Saved analysis')

    except (ValueError,
            TypeError,
            IndexError,
            FileExistsError,
            FileNotFoundError,
            ArithmeticError,
            NameError,
            LookupError,
            AssertionError,
            ) as err:
        logger.error(err.with_traceback())
