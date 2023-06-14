"""Module containing a class to identify the type of a given cloud.

This module contains the class Classification which is initialised at the
beginning of the program. It returns a pandas DataFrame which contains the
probabilities of each cloud type for each cloud.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from typing import List

from src.classification.cloud_types import CLOUD_TYPES
from src.constants import ABS_PATH, analysis_features

logger = logging.getLogger('DeepCream.classification')


class Classification:
    """A class to determine the type of a cloud.

    First the class has to be initialised. This has to happen only once.
    Then the method Classification.evaluate takes a dataframe of cloud
    parameters as input (the one returned by Analysis.evaluate) and returns
    another dataframe with the columns being the different types of clouds, the
    rows the clouds and the cells the probabilities for each one.

        Attributes:
            type_columns:
            A list of names of the cloud types. These are the columns in the
            dataframe returned by Classification.evaluate.

            __scaler:
            A scikit-learn standard scaler used for standardizing the
            parameters of the clouds. It is preconfigured saved in the
            standard_scaler.bin file.
    """
    type_columns = []
    for group_name, group_ in CLOUD_TYPES.items():
        for type_name, type_ in group_.items():
            type_columns.append(': '.join([group_name, type_name]))

    def __init__(self):
        self.__scaler = joblib.load(
            os.path.join(ABS_PATH, 'DeepCream', 'classification',
                         'standard_scaler.bin'))

    def evaluate(self, analysis: pd.DataFrame):
        """ Gets the probabilities to be of a specific cloud type for each cloud

        This method calls the method __iter_over_types for each cloud and
        combines the results to a single dataframe.

        Attributes:
            analysis:
            A pandas DataFrame with the columns being the different parameters
            and the rows clouds. This has to have a specific set of parameters
            returned by the evaluate method of the class Analysis.

        Returns:
        Dataframe with the columns being the different types of clouds, the
        rows the clouds and the cells the probabilities for each one.
        """

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

    def __iter_over_types(self, cloud: np.ndarray) -> List[float]:
        """Method for calling the method __check_type for each cloud type and
        aggregating the results.

        This method iterates over all subtypes in CLOUD_TYPES and appends the
        resulting error to a list. Then to this list the functions e^-x is
        applied to ensure that the previous space of values from 0 to infinity
        is reduced and inverted to 1 to 0. The values are then normalised so
        that their sum equals 1 and returned as a list.

        Arguments:
            cloud:
            A numpy array containing the parameters for a single cloud. This
            equals a row in the DataFrame given to evaluate.

        Returns:
            A list containing the probabilities for each cloud type. This is
            sorted in the order found in CLOUD_TYPES.
        """
        errors = []
        for group in CLOUD_TYPES.values():
            for type_ in group.values():
                errors.append(self.__check_type(type_, cloud))

        errors = np.array(errors)

        normalized = np.exp(-errors)
        probabilities = normalized / np.sum(normalized)

        return list(probabilities)

    def __check_type(self, cloud_type: dict, cloud: np.array) -> float:  # type: ignore
        """A method for comparing a cloud against a cloud type.

        This method determines how well a cloud fits to a cloud type. Note that
        properties of a cloud are not considered if the cloud type does not
        contain an ideal value for it.

            Arguments:
                cloud_type:
                A dictionary which corresponds to a cloud type. See cloud_types
                for a more detailed description of its keys and values.

                cloud:
                A numpy array containing the parameters for a single cloud.
                It equals a row of the DataFrame given to evaluate.

            Returns:
                The root mean squared error between the given ideal cloud type
                and the cloud parameters.
        """
        errors = [abs(value - cloud[analysis_features.index(key)])
                  for key, value in cloud_type.items()]
        root_mean_squared_error = np.sqrt(np.mean(np.array(errors) ** 2))
        logging.debug(f'Root mean squared error of cloud type: '
                      f'{root_mean_squared_error}')

        return root_mean_squared_error
