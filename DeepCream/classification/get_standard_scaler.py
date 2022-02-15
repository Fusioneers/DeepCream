import logging
import os
import pickle

from sklearn import preprocessing
import joblib

from DeepCream.classification.load_database import get_data_frame_from_database
from DeepCream.constants import ABS_PATH

path = os.path.join(ABS_PATH, 'DeepCream', 'classification',
                    'standard_scaler.bin')

logger = logging.getLogger('DeepCream.get_standard_scaler')

df = get_data_frame_from_database(
    os.path.join(ABS_PATH, 'data', 'test_database 2022-02-15 20-32-31'))
logger.info(f'Dataframe descriptive statistics: {df.describe()}')

data = df.to_numpy()
logger.info(f'Converted dataframe to ndarray with shape {data.shape}')

scaler = preprocessing.StandardScaler().fit(data)
logger.info(f'Created scaler {scaler}')

joblib.dump(scaler, path)
logger.info(f'Saved scaler to {path}')
