import logging
import os

import numpy as np
import pandas as pd

import joblib

from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.classification.cloud_types import CLOUD_TYPES
from DeepCream.constants import ABS_PATH

logger = logging.getLogger('DeepCream.cloud_analysis.classification')


class Classification:
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

    def __init__(self):
        self.__scaler = joblib.load(
            os.path.join(ABS_PATH, 'DeepCream', 'classification',
                         'standard_scaler.bin'))

    def get_classification(self, analysis: pd.DataFrame):
        data = analysis.to_numpy()
        scaled = self.__scaler.transform(data)

    def __iter_over_types(self, cloud: np.ndarray):
        pass

    def __check_type(self, cloud_type: dict, cloud: np.array):
        errors = [abs(value - cloud[self.columns.index(key)])
                  for key, value in cloud_type.items()]
        mean_squared_error = np.mean(np.array(errors) ** 2)
        logging.debug(
            f'Mean squared error of cloud type: {mean_squared_error}')

        return mean_squared_error
