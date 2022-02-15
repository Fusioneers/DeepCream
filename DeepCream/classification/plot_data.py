# Not working

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from DeepCream.database import DataBase
from DeepCream.constants import ABS_PATH
import os

database = DataBase(os.path.normpath(os.path.join(ABS_PATH, 'data/database')))

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
    clouds = database.load_analysis_by_id(str(identifier))
    img = database.load_orig_by_id(str(identifier))
    # i0 = int(img.shape[0] * 0.45)
    # i1 = int(img.shape[0] * 0.55)
    # i2 = int(img.shape[1] * 0.45)
    # i3 = int(img.shape[1] * 0.55)
    #
    # print(int(img.shape[0] / 0.55))
    # print(int(img.shape[0] / 0.45))
    # print(int(img.shape[1] / 0.45))
    # print(int(img.shape[1] / 0.55))
    # square = img[[i0, i1], [i2, i3]]
    # print(square)
    # print(np.mean(square))
    # if not np.mean(square) < 80:
    all_clouds = all_clouds.append(clouds)

all_clouds.columns = columns + ['number', ]
print(all_clouds.describe())
