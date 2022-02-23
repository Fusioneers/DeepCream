"""A module for creating a database to store the results of DeepCream.

This module contains the DataBase class, which gives easy access to the folder
structure in which the data gained by the DeepCream module is stored.

    Typical usage example:

    from DeepCream.database import Database

    database = Database('database')
    orig = ...
    identifier = database.save_orig(orig)
    mask = ...
    database.save_mask(identifier, mask)

    orig = database.load_orig(identifier)
"""

import json
import logging
import os
import sys
import traceback
from typing import Union

import cv2 as cv
import numpy as np
import pandas as pd

from DeepCream.constants import (get_time,
                                 MAX_DATABASE_SIZE,
                                 QUALITY_THRESHOLD,
                                 )

logger = logging.getLogger('DeepCream.database')

logger.info('Initialised database')


class DataBase:
    """A class to manage a database used for DeepCream

    The folder structure:

    base_dir/
        metadata.json
        data/
            1/
                orig.png or orig.jpg
                mask.png
                analysis.csv
                classification.csv
                pareidolia.csv
            2/
                ...
            ...

    An identifier refers to a subfolder of data.  It contains all data for a
    single taken image called orig. If orig is neither compressed nor deleted,
    it is saved as a png otherwise a jpg or not. The mask is always saved as a
    png because of the relative simple black and white image and the low size
    it used only a few kilobytes. The analysis, classification and pareidolia
    dataframes are saved as csv's (without the index column).

    In the metadata file information about the database and each identifier is
    stored. It has the following structure:

    metadata.json/
        metadata/
            creation time: The time at the creation of the database
            size: The size of the database in bytes
            num images: The number of images/identifier in the database
            num compressed images: The number of compressed images
            num deleted images: The number of deleted images
        data/
            1/
                orig creation time: The time at saving of orig
                created mask: The time at saving the mask, otherwise False
                created analysis
                created classification
                created pareidolia
                compressed: If orig was at some point in time compressed
                deleted: If orig is deleted
                quality: The quality of orig as described in the documentation
            2/
                ...
            ...


        Attributes:
            base_dir:
            The path to the database

            data_dir:
            The path to the data folder

            self.metadata:
            The metadata of the database represented as a dictionary
    """

    class OrigPrioritisationError(MemoryError):
        """An error which is raised when the database is full, but images
        cannot be reduced because images have no quality yet."""

        pass

    class DataBaseFullError(MemoryError):
        """A critical error which is raised when the database is full and
        there is no more room for compression of deletion of images."""

        pass

    def __init__(self, directory):
        """Initialises the database.

        Args:
            directory:
            The path to the database. The directory is loaded as a database if
            it contains a metadata.json file (e.g. when DeepCream is restarted)
            and created if not.

        """

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

            if self.__check_full():
                raise DataBase.DataBaseFullError(
                    f'The database is too large with size '
                    f'{self.metadata["metadata"]["size"]}')
            logger.info('Created database')

    def __update_metadata(self):
        """Updates the metadata file according to the metadata dictionary."""
        self.metadata['metadata']['size'] = self.__get_size()

        with open(os.path.join(self.base_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(self.metadata, indent=4))

    def __get_param(self, identifier: str, param: str) -> Union[
        float, str, bool]:
        """Gets the value of the parameter of an identifier."""
        return self.metadata['data'][str(identifier)][param]

    def __get_path(self, identifier: str, name: str) -> str:
        """Gets the path of a file in an identifier."""

        return os.path.join(self.data_dir, str(identifier), name)

    def __save_img(self, identifier: str, name: str, img: np.ndarray):
        """Saves an image to an identifier.

        Args:
            identifier:
            The identifier to save to.

            name:
            The name of the file.

            img:
            An image in RGB as a ndarray.
        """

        if len(img.shape) == 3:
            cv.imwrite(self.__get_path(identifier, name),
                       cv.cvtColor(img, cv.COLOR_RGB2BGR))
        else:
            cv.imwrite(self.__get_path(identifier, name), img)

    def __load_img(self, identifier: str, name: str) -> np.ndarray:
        """Loads an image from an identifier."""

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
        """Gets the current size of the database in bytes."""

        size = 0
        for dirpath, dirnames, filenames in os.walk(self.base_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)

                if not os.path.islink(file_path):
                    size += os.path.getsize(file_path)
        logger.debug(f'Database size: {size}')
        return size

    def __get_quality(self, identifier: str) -> float:
        """Gets the quality of an identifier.

        The quality determines how confident the classification and pareidolia
        with their predictions are. The quality is the sum of the squared
        highest probabilities in the respective dataframes. Note that the
        probabilities are multiplied with the number of possible answers for
        each one (so 9 different cloud types and 54 different objects) to
        ensure that the calculation is fair.

        Args:
            identifier:
                The identifier containing the classification and pareidolia of
                which the quality is computed.

        Returns:
            The quality as a value from 2 to 2997 (9**2 + 54**2) where a 2
            indicates that the classification and pareidolia have no idea
            about all of the clouds. 2997 means that the algorithms are very
            sure about at least one of clouds.
        """

        if not self.metadata['data'][identifier]['created classification']:
            raise ValueError(
                'This identifier does not contain a classification')

        if not self.metadata['data'][identifier]['created pareidolia']:
            raise ValueError('This identifier does not contain a pareidolia')

        max_classification = np.max(
            np.max(self.load_classification(identifier))) * 9

        max_pareidolia = np.max(
            np.max(self.load_pareidolia(identifier))) * 54

        return np.sqrt(max_classification ** 2 + max_pareidolia ** 2)

    def __check_full(self) -> bool:
        """Checks if the database has reached its maximum capacity."""

        return self.metadata['metadata']['size'] > MAX_DATABASE_SIZE

    def __compress_orig(self, identifier: str):
        """Compresses the orig of the provided identifier with jpg"""

        logger.debug(f'Attempting to compress image {identifier}')
        if self.metadata['data'][identifier]['quality']:

            # First the png is deleted and then as a jpg saved
            orig = self.__load_img(identifier, 'orig.png')
            os.remove(self.__get_path(identifier, 'orig.png'))
            self.__save_img(identifier, 'orig.jpg', orig)

            self.metadata['data'][identifier]['compressed'] = get_time()
            self.metadata['metadata']['num compressed images'] += 1
            self.__update_metadata()
            logger.debug(f'Compressed image {identifier}')
        else:
            raise ValueError(f'Tried to compress image {identifier} which has '
                             f'no quality')

    def delete_orig(self, identifier: str):
        """Deletes the orig of the provided identifier"""
        logger.debug(f'Attempting to delete orig {identifier}')
        if os.path.exists(self.__get_path(identifier, 'orig.jpg')):
            os.remove(self.__get_path(identifier, 'orig.jpg'))
            self.metadata['metadata']['num compressed images'] -= 1
        elif os.path.exists(self.__get_path(identifier, 'orig.png')):
            # If orig.jpg does not exist then orig.png is deleted
            os.remove(self.__get_path(identifier, 'orig.png'))
        else:
            raise ValueError(f'Tried to delete image {identifier} which '
                             f'does not exist.')
        self.metadata['data'][identifier]['deleted'] = get_time()
        self.metadata['metadata']['num deleted images'] += 1
        self.__update_metadata()
        logger.debug(f'Deleted orig {identifier}')

    def __free_space(self):
        """Frees up an identifier. See the documentation for the algorithm."""

        logger.debug('Attempting to free up space')
        not_deleted = [key for key, value in self.metadata['data'].items() if
                       value['quality'] and not value['deleted']]

        if not not_deleted:
            no_quality = [key for key, value in self.metadata['data'].items()
                          if not value['deleted']]
            if no_quality:
                raise DataBase.OrigPrioritisationError('There are too many '
                                                       'origs without quality')
            else:
                raise DataBase.DataBaseFullError(
                    'There are no not yet deleted images '
                    'available')

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

            self.delete_orig(worst_img)

        self.__update_metadata()

    def __check_space(self, orig: np.ndarray):
        """Checks if space needs to be freed."""

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
        """Creates a new identifier, checks if space needs to be freed, saves
        the provided orig in it and returns the identifier."""

        identifier = 1
        while str(identifier) in self.metadata['data']:
            identifier += 1
        identifier = str(identifier)

        logger.debug(f'Attempting to save orig to {identifier}')

        if not orig.size:
            logger.error(f'Orig {identifier} is empty')
            raise ValueError(f'Orig {identifier} is empty')

        self.__check_space(orig)

        os.mkdir(os.path.join(self.data_dir, identifier))
        self.metadata['data'][identifier] = {
            'orig creation time': get_time(),
            'created mask': False,
            'created analysis': False,
            'created classification': False,
            'created pareidolia': False,
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
        """Saves a mask to an identifier."""

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
        """Saves an analysis to an identifier."""

        logger.debug(f'Attempting to save analysis to {identifier}')
        if analysis.empty:
            logger.warning(f'Analysis trying to save in {identifier} is empty')

        if identifier not in self.metadata['data']:
            raise ValueError(f'Identifier {identifier} is not in database')

        self.metadata['data'][identifier][
            'created analysis'] = get_time()

        analysis.to_csv(self.__get_path(identifier, 'analysis.csv'),
                        index=False)

        self.__update_metadata()

        logger.info(f'Saved analysis to {identifier}')

    def save_classification(self, classification: pd.DataFrame,
                            identifier: str):
        """Saves a classification to an identifier."""

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

        if self.metadata['data'][identifier]['created pareidolia']:
            self.metadata['data'][identifier]['quality'] = self.__get_quality(
                identifier)
            self.__update_metadata()

        logger.info('Saved classification')

    def save_pareidolia(self, pareidolia: pd.DataFrame, identifier: str):
        """Saves a pareidolia to an identifier."""

        logger.debug(f'Attempting to save pareidolia to {identifier}')
        if not len(pareidolia):
            logger.warning(
                f'Pareidolia trying to save in {identifier} is empty')
        if identifier not in self.metadata['data']:
            raise ValueError('Identifier not in database')

        self.metadata['data'][identifier][
            'created pareidolia'] = get_time()

        pareidolia.to_csv(self.__get_path(identifier, 'pareidolia.csv'),
                          index=False)

        self.__update_metadata()

        if self.metadata['data'][identifier]['created classification']:
            self.metadata['data'][identifier]['quality'] = self.__get_quality(
                identifier)
            self.__update_metadata()

        logger.info('Saved pareidolia')

    def load_orig(self, identifier: str) -> np.ndarray:
        """Loads an orig from an identifier."""

        logger.debug(f'Attempting to load orig from {identifier}')
        if identifier not in self.metadata['data']:
            raise ValueError(
                f'Identifier {identifier} not available in database')

        orig = self.__load_img(identifier, 'orig.png')
        logger.info(f'Loaded orig from {identifier}')
        return orig

    def load_mask(self, identifier: str) -> np.ndarray:
        """Loads a mask from an identifier."""

        logger.debug(f'Attempting to load mask from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError(
                f'Identifier {identifier} not available in database')

        if not self.metadata['data'][identifier]['created mask']:
            raise ValueError(
                f'Identifier {identifier}  does not contain a mask')

        mask = self.__load_img(identifier, 'mask.png')
        logger.info(f'Loaded mask from {identifier}')
        return mask

    def load_analysis(self, identifier: str) -> pd.DataFrame:
        """Loads an analysis from an identifier."""

        logger.debug(f'Attempting to load analysis from {identifier}')
        if identifier not in self.metadata['data']:
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created analysis']:
            raise ValueError(
                f'Identifier {identifier} does not contain an analysis')

        analysis = pd.read_csv(self.__get_path(identifier, 'analysis.csv'))
        logger.info(f'Loaded analysis from {identifier}')
        return analysis

    def load_classification(self, identifier: str) -> pd.DataFrame:
        """Loads a classification from an identifier."""

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

    def load_pareidolia(self, identifier: str) -> pd.DataFrame:
        """Loads a pareidolia from an identifier."""

        logger.debug(f'Attempting to load pareidolia from {identifier}')
        if identifier not in self.metadata['data']:
            logger.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        if not self.metadata['data'][identifier]['created pareidolia']:
            raise ValueError(
                f'Identifier {identifier} does not contain an pareidolia')

        pareidolia = pd.read_csv(self.__get_path(identifier, 'pareidolia.csv'))

        logger.info(f'Loaded pareidolia from {identifier}')
        return pareidolia

    def load_id(self, contains: str, contains_not: str) -> Union[str, None]:
        """Loads the first identifier containing 'contains', but not
        'contains not' i.e. an identifier containing an analysis but not a
        classification."""

        out = None
        for identifier in range(1, len(self.metadata['data']) + 1):
            img = self.metadata['data'][str(identifier)]
            if not img['deleted'] and not img['compressed'] \
                    and img[contains] and not img[contains_not]:
                out = str(identifier)
                break

        return out
