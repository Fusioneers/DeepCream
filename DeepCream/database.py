import json
import logging
import os
import sys
import traceback
from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd

from DeepCream.constants import get_time, MAX_DATABASE_SIZE

logger = logging.getLogger('DeepCream.database')

logger.info('Initialised database')


# TODO docstrings
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
                    compressed
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
                logger.info('Loaded database')
            except FileNotFoundError:
                logger.error('Directory is not a DataBase')
                raise ValueError('Directory is not a DataBase')

        else:
            os.mkdir(self.base_dir)
            os.mkdir(self.data_dir)

            self.metadata = {
                'metadata': {
                    'creation time': get_time(),
                    'size': 0
                },
                'data': {},
            }

            self.__update_metadata()
            logger.info('Created database')

    def __update_metadata(self):
        self.metadata['metadata']['size'] = self.__get_size()
        with open(os.path.join(self.base_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(self.metadata, indent=4))

    def __get_id_path(self, identifier: str) -> str:
        return os.path.join(self.data_dir, str(identifier))

    def __get_param(self, identifier: str, param: str):
        return self.metadata['data'][str(identifier)][param]

    def __get_path(self, identifier: str, name: str):
        return os.path.join(self.__get_id_path(str(identifier)), name)

    def __save_img(self, identifier: str, name: str, img: np.ndarray):
        cv.imwrite(self.__get_path(identifier, name),
                   cv.cvtColor(img, cv.COLOR_RGB2BGR))

    def __load_img(self, identifier: str, name: str) -> np.ndarray:
        img = cv.cvtColor(
            cv.imread(self.__get_path(identifier, name)),
            cv.COLOR_BGR2RGB)
        try:
            img.size
        except (ValueError, AttributeError):
            logger.error(f'Could not load {name}')
        logger.info('Loaded img')
        return img

    def __get_size(self) -> int:
        size = 0
        for dirpath, dirnames, filenames in os.walk(self.base_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)

                if not os.path.islink(file_path):
                    size += os.path.getsize(file_path)
        logger.debug(f'Database size: {size}')
        return size

    def __get_quality(self, identifier: str) -> int:
        return np.random.random()

    def __compress_orig(self, identifier: str):
        logger.debug('Attempting to compress orig')
        if self.metadata['data'][identifier]['quality']:
            try:
                orig = self.__load_img(identifier, 'orig.png')
                os.remove(self.__get_path(identifier, 'orig.png'))
                self.__save_img(identifier, 'orig.jpg', orig)

                self.metadata['data'][identifier]['compressed'] = get_time()
                self.__update_metadata()
                logger.debug(f'Compressed orig {identifier}')
            except (OSError, ValueError) as err:
                logger.error(traceback.format_exc())
                raise err
        else:
            logger.error(
                'Tried to compress image which was not yet classified')
            raise ValueError(
                'Tried to compress image which was not yet classified')

    def __delete_orig(self, identifier: str):
        logger.debug('Attempting to compress orig')
        if self.metadata['data'][identifier]['quality']:
            try:
                os.remove(self.__get_path(identifier, 'orig.jpg'))
                self.metadata['data'][identifier]['deleted'] = get_time()
                self.__update_metadata()
                logger.debug(f'Deleted orig {identifier}')
            except (OSError, ValueError) as err:
                logger.error(traceback.format_exc())
                raise err
        else:
            logger.error('Tried to delete image which was not yet classified')
            raise ValueError(
                'Tried to delete image which was not yet classified')

    def __free_space(self):
        logger.debug('Attempting to free up space')
        not_deleted = [key for key, value in self.metadata['data'].items() if
                       value['quality'] and not value['deleted']]

        # TODO this case
        if not not_deleted:
            logging.error(
                'There are no evaluated and not yet deleted images available')
            raise MemoryError(
                'There are no evaluated and not yet deleted images available')

        # TODO implement quality_threshold
        # TODO maybe warning when deleting/compressing for the first time

        not_compressed = list(
            filter(lambda x: not self.__get_param(x, 'compressed'),
                   not_deleted))

        if not_compressed:
            qualities = list(map(lambda x: self.__get_param(x, 'quality'),
                                 not_compressed))
            worst_img_idx = np.argmin(np.array(qualities))
            worst_img = not_compressed[worst_img_idx]

            self.__compress_orig(worst_img)
            logger.info(f'Compressed orig {worst_img}')
        else:
            qualities = list(map(lambda x: self.__get_param(x, 'quality'),
                                 not_deleted))
            worst_img_idx = np.argmin(np.array(qualities))
            worst_img = not_deleted[worst_img_idx]

            self.__delete_orig(worst_img)
            logger.debug(f'Deleted orig {worst_img}')

        self.__update_metadata()

    def __check_space(self, orig: np.ndarray):
        is_success, orig_png = cv.imencode('.png', orig)
        if not is_success:
            logger.error('Failed converting ndarray to png')
        orig_png_size = sys.getsizeof(orig_png)
        logger.debug(f'orig size: {orig_png_size}')

        if self.metadata['metadata']['size'] \
                + orig_png_size >= MAX_DATABASE_SIZE:
            logger.debug('Attempting to free up space')
            logger.debug(f'Database size: {self.metadata["metadata"]["size"]}')
            while self.metadata['metadata']['size'] \
                    + orig_png_size >= MAX_DATABASE_SIZE:
                self.__free_space()
            logger.info('Freed up space')

    def save_orig(self, orig: np.ndarray) -> str:
        logger.debug('Attempting to save orig')

        if not orig.size:
            logger.error('Orig is empty')
            raise ValueError('Orig is empty')

        self.__check_space(orig)

        identifier = 1
        while str(identifier) in self.metadata['data']:
            identifier += 1
        identifier = str(identifier)

        os.mkdir(self.__get_id_path(identifier))
        self.metadata['data'][identifier] = {
            'orig creation time': get_time(),
            'created mask': False,
            'created analysis': False,
            'created classification': False,
            'compressed': False,
            'deleted': False,
            'quality': None
        }

        self.__save_img(identifier, 'orig.png', orig)

        logger.info('Saved orig')
        self.__update_metadata()

        return identifier

    def save_mask(self, mask: np.ndarray, identifier: str):
        if not mask.size:
            logger.error('Mask is empty')
            raise ValueError('Mask is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier]['created mask'] = get_time()

        self.__save_img(identifier, 'mask.png', mask)
        self.__update_metadata()

        logger.info('Saved mask')

    def save_analysis(self, analysis: pd.DataFrame, identifier: str):
        if analysis.empty:
            logger.error('Analysis is empty')
            raise ValueError('Analysis is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created analysis'] = get_time()

        analysis.to_csv(self.__get_path(identifier, 'analysis.csv'))
        self.__update_metadata()

        # TODO this is just temporary for testing, remove for actual program!
        self.metadata['data'][identifier]['quality'] = self.__get_quality(
            identifier)

        logger.info('Saved analysis')

    def save_classification(self, classification: pd.DataFrame,
                            identifier: str):
        if classification.empty:
            logger.error('Classification is empty')
            raise ValueError('Classification is empty')

        if identifier not in self.metadata['data']:
            logger.error('Identifier not in database')
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created classification'] = get_time()

        classification.to_csv(
            self.__get_path(identifier, 'classification.csv'))
        self.__update_metadata()

        self.metadata['data'][identifier]['quality'] = self.__get_quality(
            identifier)

        logger.info('Saved classification')

    def load_orig_by_id(self, identifier: str) -> np.ndarray:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        return self.__load_img(identifier, 'orig.png')

    def load_orig_by_empty_mask(self) -> Tuple[np.ndarray, str]:
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

        return pd.read_csv(self.__get_path(identifier, 'analysis.csv'))

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

        return (
            pd.read_csv(self.__get_path(identifier, 'orig.png')), identifier)

    def load_classification_by_id(self, identifier: str) -> pd.DataFrame:
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created classification']:
            logger.error('Identifier does not contain an classification')
            raise ValueError('Identifier does not contain an classification')

        return pd.read_csv(self.__get_path(identifier, 'classification.csv'))
