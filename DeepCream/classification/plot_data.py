import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from DeepCream.database import DataBase
from DeepCream.constants import ABS_PATH
import os

database = DataBase(os.path.normpath(os.path.join(ABS_PATH, 'data/database')))
columns = ['center_x',
           'center_y',
           'contour_perimeter',
           'contour_area',
           'hull_perimeter',
           'hull_area',
           'rectangularity',
           'elongation',
           'mean_r',
           'mean_g',
           'mean_b',
           'std_r',
           'std_g',
           'std_b',
           'transparency',
           'mean_diff_edges',
           ]

columns = ['center_x',
           'center_y',
           'contour_perimeter',
           'contour_area',
           'hull_perimeter',
           'hull_area',
           'rectangularity',
           'elongation',
           'mean_r',
           'mean_g',
           'mean_b',
           'std_r',
           'std_g',
           'std_b',
           'transparency',
           'mean_diff_edges',
           ]

all_clouds = pd.DataFrame(columns=columns)
for identifier in range(1, 250):
    clouds = database.load_analysis_by_id(str(identifier))
    img = database.load_orig_by_id(str(identifier))
    i0 = int(img.shape[0] * 0.45)
    i1 = int(img.shape[0] * 0.55)
    i2 = int(img.shape[1] * 0.45)
    i3 = int(img.shape[1] * 0.55)
    #
    # print(int(img.shape[0] / 0.55))
    # print(int(img.shape[0] / 0.45))
    # print(int(img.shape[1] / 0.45))
    # print(int(img.shape[1] / 0.55))
    square = img[[i0, i1], [i2, i3]]
    # print(square)
    # print(np.mean(square))
    if not np.mean(square) < 80:
        all_clouds = all_clouds.append(clouds)

all_clouds.columns = columns + ['number', ]
print(all_clouds.describe())

# sb.pairplot(all_clouds)  # , hue='Unnamed: 0')
# plt.show()
g = sb.PairGrid(all_clouds)


def scatter_fake_diag(x, y, *a, **kw):
    if x.equals(y):
        kw["color"] = (0, 0, 0, 0)
    plt.scatter(x, y)  # , kind='kde', hue='Unnamed: 0')


g.map(scatter_fake_diag)
g.map_diag(plt.hist)
plt.show()
