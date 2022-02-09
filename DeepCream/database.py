import json
import logging
import os

import numpy as np
import cv2 as cv

from utils import get_time


class DataBase:
    # TODO document the structure of the database

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
                    orig_creation_time
                    mask_creation_time
                    analysis_creation_time
                    type_creation_time
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
                        type

    """

    def __init__(self, directory):
        self.base_dir = directory
        if os.path.isdir(self.base_dir):
            if not any(os.scandir(self.base_dir)):
                logging.error('Directory is not empty')
                raise ValueError('Directory is not empty')
        else:
            logging.error('Path is not a directory')
            raise ValueError('Path is not a directory')

        self.metadata = {
            'metadata': {
                'Creation time': get_time(),
            },
            'data': {},
        }

        self.__update_metadata_file()
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.mkdir(self.data_dir)

    def __update_metadata_file(self):
        with open(os.path.join(self.base_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(self.metadata, indent=4))

    def __get_id_path(self, identifier: int) -> str:
        return os.path.join(self.data_dir, str(identifier))

    def save_orig(self, orig: np.ndarray) -> int:
        identifier = len(self.metadata['data']) + 1
        # TODO if identifier in database (shouldn't happen, but who knows...)

        os.mkdir(os.path.join(self.data_dir, str(identifier)))
        self.metadata['data'][identifier] = {
            'orig_creation_time': get_time(),
            'mask_creation_time': None,
            'analysis_creation_time': None,
            'type_creation_time': None,
            'is_compressed': False,
        }

        cv.imwrite(
            os.path.join(self.__get_id_path(identifier), 'orig.png'), orig)
        logging.info('Saved orig')

        self.__update_metadata_file()

        return identifier

    def load_orig_by_id(self, identifier: int) -> np.ndarray:
        if identifier not in self.metadata['data']:
            logging.error('Identifier not available in database')
            raise ValueError('Identifier not available in database')

        return cv.imread(
            os.path.join(self.data_dir, str(identifier), 'orig.png'))

    def load_orig_by_empty_mask(self) -> np.ndarray:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['mask_creation_time']:
                    identifier = img
                    break

        if not identifier:
            logging.error('No not masked image in database')
            raise LookupError('No not masked image in database')

        return cv.imread(
            os.path.join(self.data_dir, str(identifier), 'orig.png'))

    def load_orig_by_empty_analysis(self) -> np.ndarray:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['analysis_creation_time']:
                    identifier = img
                    break

        if not identifier:
            logging.error('No not analysed image in database')
            raise LookupError('No not analysed image in database')

        return cv.imread(
            os.path.join(self.data_dir, str(identifier), 'orig.png'))

    def load_orig_by_empty_type(self) -> np.ndarray:
        identifier = None
        for img in range(1, len(self.metadata['data']) + 1):
            if img in self.metadata['data']:
                if not self.metadata['data'][img]['type_creation_time']:
                    identifier = img
                    break

        if not identifier:
            logging.error('No not analysed image in database')
            raise LookupError('No not analysed image in database')

        return cv.imread(
            os.path.join(self.data_dir, str(identifier), 'orig.png'))
