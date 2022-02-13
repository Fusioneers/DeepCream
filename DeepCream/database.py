import json
import logging
import os
from typing import Tuple, Any

import cv2 as cv
import numpy as np
import pandas as pd

from DeepCream.constants import DEFAULT_COMPRESSION_DIM, get_time

logger = logging.getLogger('DeepCream.database')


# TODO docstrings
# TODO tests
class DataBase:
    """
    
Structure of the database:

    base/
        metadata.json/
            creation time
            folder structure/
                this text
        data/
            1/
                metadata.json/
                    orig creation time
                    created mask
                    created analysis
                    created classification
                    is_compressed
                orig.png or compressed.jpg
                mask.png
                analysis.csv/
                    1/ (cloud)
                        center_x
                        center_y
                        contour_perimeter
                        contour_area
                        hull_perimeter
                        hull_area
                        rectangularity
                        elongation
                        mean_r
                        mean_g
                        mean_b
                        std_r
                        std_g
                        std_b
                        transparency
                        mean_diff_edges
                classification.csv/
                    1/ (cloud)
                        center_x
                        center_y
                        type1
                        type2
                        ...

    """

    def __init__(self, directory):
        self.base_dir = directory
        self.data_dir = os.path.join(self.base_dir, 'data')
        if os.path.isdir(self.base_dir):
            try:
                with open(os.path.join(self.base_dir, 'metadata.json'),
                          'r') as metadata:
                    self.metadata = json.load(metadata)
            except FileNotFoundError:
                logger.error('Directory is not a DataBase')
                raise ValueError('Directory is not a DataBase')

        else:
            os.mkdir(self.base_dir)
            os.mkdir(self.data_dir)

            self.metadata = {
                'metadata': {
                    'Creation time': get_time(),
                },
                'data': {},
            }

            self.__update_metadata_file()

    def __update_metadata_file(self):
        with open(os.path.join(self.base_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(self.metadata, indent=4))

    def __get_id_path(self, identifier: str) -> str:
        return os.path.join(self.data_dir, identifier)

    def __save_img(self, identifier: str, name: str, img: np.ndarray):
        cv.imwrite(os.path.join(self.__get_id_path(identifier), name),
                   cv.cvtColor(img, cv.COLOR_RGB2BGR))

    def __load_img(self, identifier: str, name: str) -> np.ndarray:
        img = cv.cvtColor(
            cv.imread(os.path.join(self.data_dir, identifier, name)),
            cv.COLOR_BGR2RGB)
        if not img:
            logger.error(f'Could not load {name}')
        return img

    def save_orig(self, orig: np.ndarray, is_compressed: bool = False) -> str:
        logger.debug('Attempting to save orig')

        if not orig.size:
            logging.error('Orig is empty')
            raise ValueError('Orig is empty')

        identifier = 1
        while str(identifier) in self.metadata['data']:
            identifier += 1
        identifier = str(identifier)

        os.mkdir(os.path.join(self.data_dir, identifier))
        self.metadata['data'][identifier] = {
            'orig creation time': get_time(),
            'created mask': False,
            'created analysis': False,
            'created classification': False,
            'is_compressed': is_compressed,
        }

        if is_compressed:
            orig = cv.resize(orig, DEFAULT_COMPRESSION_DIM)

        logger.info('Saved orig')

        self.__save_img(identifier, 'orig.png', orig)
        self.__update_metadata_file()

        return identifier

    def save_mask(self, mask: np.ndarray, identifier: str):
        if not mask.size:
            logging.error('Mask is empty')
            raise ValueError('Mask is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier]['created mask'] = get_time()

        self.__save_img(identifier, 'mask.png', mask)
        self.__update_metadata_file()

        logger.info('Saved mask')

    def save_analysis(self, analysis: pd.DataFrame, identifier: str):
        if analysis.empty:
            logging.error('Analysis is empty')
            raise ValueError('Analysis is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created analysis'] = get_time()

        analysis.to_csv(
            os.path.join(self.__get_id_path(identifier), 'analysis.csv'))
        self.__update_metadata_file()

        logger.info('Saved analysis')

    def save_classification(self, classification: pd.DataFrame,
                            identifier: str):
        if classification.empty:
            logging.error('Classification is empty')
            raise ValueError('Classification is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created classification'] = get_time()

        classification.to_csv(
            os.path.join(self.__get_id_path(identifier), 'classification.csv'))
        self.__update_metadata_file()

        logger.info('Saved classification')

    def load_orig_by_id(self, identifier: str) -> np.ndarray:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        return self.__load_img(identifier, 'orig.png')

    def load_orig_by_empty_mask(self) -> Tuple[np.ndarray, int]:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created mask']:
                    identifier = img
                    break

        if not identifier:
            logger.error('No not masked image in database')
            raise LookupError('No not masked image in database')

        return self.__load_img(identifier, 'orig.png'), identifier

    def load_orig_by_empty_analysis(self) -> Tuple[np.ndarray, str]:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created analysis']:
                    identifier = img
                    break

        if not identifier:
            logger.error('No not analysed image in database')
            raise LookupError('No not analysed image in database')

        return self.__load_img(identifier, 'orig.png'), identifier

    def load_mask_by_id(self, identifier: str) -> np.ndarray:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created mask']:
            logger.error('Identifier does not contain a mask')
            raise ValueError('Identifier does not contain a mask')

        return self.__load_img(identifier, 'mask.png')

    def load_mask_by_empty_analysis(self) -> Tuple[np.ndarray, str]:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created analysis'] and \
                        self.metadata['data'][identifier]['created mask']:
                    identifier = img
                    break

        if not identifier:
            logger.error('No not analysed image in database')
            raise LookupError('No not analysed image in database')

        return self.__load_img(identifier, 'mask.png'), identifier

    def load_analysis_by_id(self, identifier: str) -> pd.DataFrame:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created analysis']:
            logger.error('Identifier does not contain an analysis')
            raise ValueError('Identifier does not contain an analysis')

        return pd.read_csv(
            os.path.join(self.data_dir, str(identifier), 'analysis.csv'))

    def load_analysis_by_empty_classification(self) \
            -> Tuple[pd.DataFrame, str]:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created classification']:
                    if self.metadata['data'][identifier]['created analysis']:
                        identifier = img
                        break

        if not identifier:
            logger.error('No not classified image in database')
            raise LookupError('No not classified image in database')

        return (pd.read_csv(os.path.join(self.data_dir, str(identifier),
                                         'orig.png')), identifier)

    def load_classification_by_id(self, identifier: str) -> pd.DataFrame:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created classification']:
            logger.error('Identifier does not contain an classification')
            raise ValueError('Identifier does not contain an classification')

        return pd.read_csv(
            os.path.join(self.data_dir, str(identifier), 'classification.csv'))
