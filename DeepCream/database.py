import os

import numpy as np


class DataStructure:

    def __init__(self, directory):
        self.base_dir = directory

    def __create_metadata(self):
        pass

    def __create_dirs(self):
        os.mkdir(os.path.join(self.base_dir + 'data'))

    def save_orig(self, img: np.ndarray):
        pass

    def save_compressed(self, img: np.ndarray):
        pass

    def save_mask(self, img: np.ndarray):
        pass

    def save_orig_data(self, data):
        pass

    def save_compressed_data(self, data):
        pass

    def save_mask_data(self, data):
        pass

    def load_orig(self, name: str) -> np.ndarray:
        pass

    def load_compressed(self, name: str) -> np.ndarray:
        pass

    def load_mask(self, name: str) -> np.ndarray:
        pass

    def load_orig_data(self, name: str):
        pass

    def load_compressed_data(self, name: str):
        pass

    def load_mask_data(self, name: str):
        pass


"""

base/
    metadata.txt/
        creation time
        folder structure/
            this text
    data/
        1/
            metadata.txt/
                orig_creation_time
                mask_creation_time
                analysis_creation_time
                interpretation_time
                is_compressed
            orig.png or compressed.jpg
            mask.png
            analysis.csv/
                1/
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
