import json
import logging
import os
import sys
import traceback
from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd

from DeepCream.constants import (DEBUG_MODE,
                                 get_time,
                                 MAX_DATABASE_SIZE,
                                 QUALITY_THRESHOLD,
                                 )

logger = logging.getLogger('DeepCream.database')

logger.info('Initialised database')


# TODO docstrings
class DataBase:

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
                    'size': 0,
                    'num images': 0,
                    'num compressed images': 0,
                    'num deleted images': 0,
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
        if len(img.shape) == 3:
            cv.imwrite(self.__get_path(identifier, name),
                       cv.cvtColor(img, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(self.__get_path(identifier, name), img)

    def __load_img(self, identifier: str, name: str) -> np.ndarray:
        img = cv.cvtColor(
            cv.imread(self.__get_path(identifier, name)),
            cv.COLOR_BGR2RGB)
        try:
            img.size
        except (ValueError, AttributeError):
            logger.error(f'Could not load {name} from image {identifier}')
        logger.debug(f'Loaded {name} from image {identifier}')
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
        logger.debug(f'Attempting to compress image {identifier}')
        if self.metadata['data'][identifier]['quality']:
            try:
                orig = self.__load_img(identifier, 'orig.png')
                os.remove(self.__get_path(identifier, 'orig.png'))
                self.__save_img(identifier, 'orig.jpg', orig)

                self.metadata['data'][identifier]['compressed'] = get_time()
                self.metadata['metadata']['num compressed images'] += 1
                self.__update_metadata()
                logger.debug(f'Compressed image {identifier}')
            except (OSError, ValueError) as err:
                logger.error(traceback.format_exc())
                raise err
        else:
            logger.error(f'Tried to compress image {identifier} which has no '
                         f'quality')
            raise ValueError(f'Tried to compress image {identifier} which has '
                             f'no quality')

    def __delete_orig(self, identifier: str):
        logger.debug(f'Attempting to delete orig {identifier}')
        if self.__get_param(identifier, 'quality'):
            if self.__get_param(identifier, 'compressed'):
                try:
                    os.remove(self.__get_path(identifier, 'orig.jpg'))
                    self.metadata['data'][identifier]['deleted'] = get_time()
                    self.metadata['metadata']['num deleted images'] += 1
                    self.metadata['metadata']['num compressed images'] -= 1
                    self.__update_metadata()
                    logger.debug(f'Deleted orig {identifier}')
                except FileNotFoundError:
                    logger.error(traceback.format_exc())
                    if os.path.exists(self.__get_path(identifier, 'orig.png')):
                        logger.error(
                            f'Tried to delete image {identifier} which is not '
                            f'compressed')
                        logger.info(
                            f'Attempting to compress image {identifier}')
                        self.__compress_orig(identifier)
                    else:
                        logger.error(
                            f'Tried to delete image {identifier} which '
                            f'was already deleted')
            else:
                logger.warning(f'Tried to delete image {identifier} which is '
                               f'not compressed, attempting to compress image '
                               f'{identifier}')
                self.__compress_orig(identifier)
        else:
            logger.error(f'Tried to delete image {identifier} '
                         f'which has no quality')
            raise ValueError(f'Tried to delete image {identifier} '
                             f'which has no quality')

    def __free_space(self):
        logger.debug('Attempting to free up space')
        not_deleted = [key for key, value in self.metadata['data'].items() if
                       value['quality'] and not value['deleted']]

        # TODO this case
        if not not_deleted:
            logger.critical(
                'There are no evaluated and not yet deleted images available')
            raise MemoryError(
                'There are no evaluated and not yet deleted images available')

        not_compressed = list(
            filter(lambda x: not self.__get_param(x, 'compressed'),
                   not_deleted))

        num_not_compressed = len(not_compressed)
        num_not_deleted = len(self.metadata['data']) \
                          - self.metadata['metadata']['num deleted images']

        if not_compressed and \
                num_not_compressed / num_not_deleted > QUALITY_THRESHOLD:
            qualities = list(map(lambda x: self.__get_param(x, 'quality'),
                                 not_compressed))
            worst_img_idx = np.argmin(np.array(qualities))
            worst_img = not_compressed[worst_img_idx]

            self.__compress_orig(worst_img)
        else:
            compressed = list(
                filter(lambda x: self.__get_param(x, 'compressed'),
                       not_deleted))

            qualities = list(map(lambda x: self.__get_param(x, 'quality'),
                                 compressed))
            worst_img_idx = np.argmin(np.array(qualities))
            worst_img = not_deleted[worst_img_idx]

            self.__delete_orig(worst_img)

        self.__update_metadata()

    def __check_space(self, orig: np.ndarray):
        is_success, orig_png = cv.imencode('.png', orig)
        if not is_success:
            logger.error('Failed converting ndarray to png')
        orig_png_size = sys.getsizeof(orig_png)
        logger.debug(f'orig size: {orig_png_size}')

        if self.metadata['metadata']['size'] \
                + orig_png_size >= MAX_DATABASE_SIZE:
            if self.metadata['metadata']['num compressed images'] \
                    or self.metadata['metadata']['num deleted images']:
                logger.debug('Attempting to free up space')
            else:
                logger.warning(
                    'The database is too large, attempting to free space')
            logger.debug(f'Database size: {self.metadata["metadata"]["size"]}')

            while self.metadata['metadata']['size'] \
                    + orig_png_size >= MAX_DATABASE_SIZE:
                self.__free_space()
            logger.info('Freed up space')

    def save_orig(self, orig: np.ndarray) -> str:
        identifier = 1
        while str(identifier) in self.metadata['data']:
            identifier += 1
        identifier = str(identifier)

        logger.debug(f'Attempting to save orig to {identifier}')

        if not orig.size:
            logger.error(f'Orig {identifier} is empty')
            raise ValueError(f'Orig {identifier} is empty')

        self.__check_space(orig)

        os.mkdir(self.__get_id_path(identifier))
        self.metadata['data'][identifier] = {
            'orig creation time': get_time(),
            'created mask': False,
            'created analysis': False,
            'created classification': False,
            'created paradolia': False,
            'compressed': False,
            'deleted': False,
            'quality': None
        }
        self.metadata['metadata']['num images'] += 1
        self.__update_metadata()

        self.__save_img(identifier, 'orig.png', orig)
        logger.info(f'Saved orig to {identifier}')

        return identifier

    def save_mask(self, mask: np.ndarray, identifier: str):
        logger.debug(f'Attempting to save mask to {identifier}')
        if not mask.size:
            raise ValueError(f'Mask in {identifier} is empty')

        if identifier not in self.metadata['data']:
            raise ValueError(f'Identifier {identifier} is not in database')

        self.metadata['data'][identifier]['created mask'] = get_time()

        self.__save_img(identifier, 'mask.png', mask)
        self.__update_metadata()

        logger.info(f'Saved mask to {identifier}')

    def save_analysis(self, analysis: pd.DataFrame, identifier: str):
        logger.debug(f'Attempting to save analysis to {identifier}')
        if analysis.empty:
            logger.warning(f'Analysis trying to save in {identifier} is empty')

        if identifier not in self.metadata['data']:
            raise ValueError(f'Identifier {identifier} is not in database')

        self.metadata['data'][identifier][
            'created analysis'] = get_time()

        analysis.to_csv(self.__get_path(identifier, 'analysis.csv'),
                        index=False)

        if DEBUG_MODE:
            self.metadata['data'][identifier]['quality'] = self.__get_quality(
                identifier)

        self.__update_metadata()

        logger.info(f'Saved analysis to {identifier}')

    def save_classification(self, classification: pd.DataFrame,
                            identifier: str):
        logger.debug(f'Attempting to save classification to {identifier}')
        if classification.empty:
            logger.warning(
                f'Classification trying to save in {identifier} is empty')

        if identifier not in self.metadata['data']:
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created classification'] = get_time()

        classification.to_csv(
            self.__get_path(identifier, 'classification.csv'),
            index=False)

        self.__update_metadata()

        logger.info('Saved classification')

    def save_paradolia(self, paradolia: dict, identifier: str):
        logger.debug(f'Attempting to save paradolia to {identifier}')
        if not len(paradolia):
            logger.warning(
                f'Paradolia trying to save in {identifier} is empty')
        if identifier not in self.metadata['data']:
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created paradolia'] = get_time()

        with open(self.__get_path(identifier, 'paradolia.json'), 'w') as f:
            f.write(json.dumps(paradolia, indent=4))

        self.metadata['data'][identifier]['quality'] = self.__get_quality(
            identifier)
        self.__update_metadata()

        logger.info('Saved paradolia')

    def load_orig_by_id(self, identifier: str) -> np.ndarray:
        logger.debug(f'Attempting to load orig to {identifier}')
        if identifier not in self.metadata['data']:
            raise ValueError(
                f'Identifier {identifier} not available in database')

        orig = self.__load_img(identifier, 'orig.png')
        logger.info(f'Loaded orig from {identifier}')
        return orig

    def load_orig_by_empty_mask(self) -> Tuple[np.ndarray, str]:
        logger.debug('Attempting to load orig by empty mask')
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created mask']:
                    identifier = img
                    break

        if not identifier:
            raise LookupError('No not masked image in database')

        orig = self.__load_img(identifier, 'orig.png')
        logger.info(f'Loaded orig by empty mask from {identifier}')
        return orig, identifier

    def load_orig_by_empty_analysis(self) -> Tuple[np.ndarray, str]:
        logger.debug('Attempting to load orig by empty analysis')
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created analysis']:
                    identifier = img
                    break

        if not identifier:
            raise LookupError('No not analysed image in database')

        orig = self.__load_img(identifier, 'orig.png')
        logger.info(f'Loaded orig by empty analysis from {identifier}')
        return orig, identifier

    def load_mask_by_id(self, identifier: str) -> np.ndarray:
        logger.debug(f'Attempting to load mask from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created mask']:
            raise ValueError('Identifier does not contain a mask')

        mask = self.__load_img(identifier, 'mask.png')
        logger.info(f'Loaded mask from {identifier}')
        return mask

    def load_mask_by_empty_analysis(self) -> Tuple[np.ndarray, str]:
        logger.debug('Attempting to load mask by empty analysis')
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created analysis'] and \
                        self.metadata['data'][identifier]['created mask']:
                    identifier = img
                    break

        if not identifier:
            raise LookupError('No not analysed image in database')

        mask = self.__load_img(identifier, 'mask.png')
        logger.info(f'Loaded mask by empty analysis from {identifier}')
        return mask, identifier

    def load_mask_by_empty_paradolia(self) -> Tuple[np.ndarray, str]:
        logger.debug('Attempting to load mask by empty paradolia')
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created paradolia'] and \
                        self.metadata['data'][identifier]['created mask']:
                    identifier = img
                    break

        if not identifier:
            raise LookupError('No not paradoliaistically interpreted image in '
                              'database')

        mask = self.__load_img(identifier, 'mask.png')
        logger.info(f'Loaded mask by empty paradolia from {identifier}')
        return mask, identifier

    def load_analysis_by_id(self, identifier: str) -> pd.DataFrame:
        logger.debug(f'Attempting to load analysis from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created analysis']:
            raise ValueError(
                f'Identifier {identifier} does not contain an analysis')

        analysis = pd.read_csv(self.__get_path(identifier, 'analysis.csv'))
        logger.info(f'Loaded analysis from {identifier}')
        return analysis

    def load_analysis_by_empty_classification(self) \
            -> Tuple[pd.DataFrame, str]:
        logger.debug('Attempting to load analysis by empty classification')
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            img = str(img)
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['created classification']:
                    if self.metadata['data'][identifier]['created analysis']:
                        identifier = img
                        break

        if not identifier:
            raise LookupError('No not classified image in database')

        analysis = pd.read_csv(self.__get_path(identifier, 'orig.png'))
        logger.info(
            f'Loaded analysis by empty classification from {identifier}')
        return analysis, identifier

    def load_classification_by_id(self, identifier: str) -> pd.DataFrame:
        logger.debug(f'Attempting to load classification from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created classification']:
            raise ValueError(
                f'Identifier {identifier} does not contain an classification')

        classification = pd.read_csv(
            self.__get_path(identifier, 'classification.csv'))
        logger.info(f'Loaded classification from {identifier}')
        return classification

    def load_paradolia_by_id(self, identifier: str) -> pd.DataFrame:
        logger.debug(f'Attempting to load paradolia from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created paradolia']:
            raise ValueError(
                f'Identifier {identifier} does not contain an paradolia')

        with open(self.__get_path(identifier, 'paradolia.json'),
                  'r') as paradolia:
            paradolia = json.load(paradolia)

        logger.info(f'Loaded paradolia from {identifier}')
        return paradolia
