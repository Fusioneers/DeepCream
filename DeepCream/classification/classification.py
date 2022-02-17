import logging
import os

import joblib
import numpy as np
import pandas as pd

from DeepCream.classification.cloud_types import CLOUD_TYPES
from DeepCream.constants import ABS_PATH, analysis_features

logger = logging.getLogger('DeepCream.classification')


class Classification:
    type_columns = []
    for group_name, group_ in CLOUD_TYPES.items():
        for type_name, type_ in group_.items():
            for sub_type_name, sub_type in type_.items():
                type_columns.append(
                    ': '.join([group_name, type_name, sub_type_name]))

    def __init__(self):
        self.__scaler = joblib.load(
            os.path.join(ABS_PATH, 'DeepCream', 'classification',
                         'standard_scaler.bin'))

    def get_classification(self, analysis: pd.DataFrame):
        data = analysis.to_numpy()
        scaled = self.__scaler.transform(data)
        probabilities = []
        for cloud in scaled:
            probability = self.__iter_over_types(cloud)
            probabilities.append(probability)

        classification = pd.DataFrame(columns=self.type_columns,
                                      data=probabilities)
        logger.info('Created classification')

        return classification

    def __iter_over_types(self, cloud: np.ndarray):
        errors = []
        for group in CLOUD_TYPES.values():
            for type_ in group.values():
                for subtype in type_.values():
                    errors.append(self.__check_type(subtype, cloud))

        errors = np.array(errors)

        normalized = np.exp(-errors)
        probabilities = normalized / np.sum(normalized)

        return list(probabilities)

    def __check_type(self, cloud_type: dict, cloud: np.array):
        errors = [abs(value - cloud[analysis_features.index(key)])
                  for key, value in cloud_type.items()]
        root_mean_squared_error = np.sqrt(np.mean(np.array(errors) ** 2))
        logging.debug(f'Root mean squared error of cloud type: '
                      f'{root_mean_squared_error}')

        return root_mean_squared_error
