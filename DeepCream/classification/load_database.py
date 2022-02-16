import pandas as pd
from DeepCream.database import DataBase
import os
import logging

logger = logging.getLogger('DeepCream.get_standard_scaler')


def get_data_frame_from_database(database: str) -> pd.DataFrame:
    database = DataBase(
        os.path.normpath(database))

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

    all_clouds = pd.DataFrame(columns=columns)
    for identifier in range(1, 250):
        try:
            clouds = database.load_analysis_by_id(str(identifier))
            all_clouds = all_clouds.append(clouds)
        except ValueError as err:
            logger.error(err)
    logger.info('Loaded dataframe from database')

    return all_clouds
