import os
from datetime import datetime
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from analysis import Analysis

path = os.path.realpath(__file__).removesuffix(
    r'cloud_analysis\save_analysis_data.py')

directory = path + r'sample_data\set_1'
num_clouds = 5

num_img = len(os.listdir(directory))

columns = ['image',
           'contour_area',
           'contour_perimeter',
           'hull_area',
           'hull_perimeter',
           'rectangularity',
           'elongation',
           'mean_r',
           'mean_g',
           'mean_b',
           'std_r',
           'std_g',
           'std_b',
           'transparency',
           'mean_diff_edges_r',
           'mean_diff_edges_g',
           'mean_diff_edges_b', ]
df = pd.DataFrame(columns=columns)

for i, img in tqdm(enumerate(os.scandir(directory)), total=num_img):
    analysis = Analysis(cv.imread(img.path), num_clouds)

    for j, cloud in enumerate(analysis.clouds):
        loc = num_clouds * i + j

        std = cloud.texture.std()
        mean = cloud.texture.mean()
        mean_diff_edges = cloud.mean_diff_edges(10, 50, 500)

        df.loc[loc, ['image']] = img.name
        df.loc[loc, ['contour_area']] = cloud.shape.contour_area
        df.loc[loc, ['contour_perimeter']] = cloud.shape.contour_perimeter
        df.loc[loc, ['hull_area']] = cloud.shape.hull_area
        df.loc[loc, ['hull_perimeter']] = cloud.shape.hull_perimeter
        df.loc[loc, ['rectangularity']] = cloud.shape.rectangularity()
        df.loc[loc, ['elongation']] = cloud.shape.elongation()

        df.loc[loc, ['mean_r']] = mean[0]
        df.loc[loc, ['mean_g']] = mean[1]
        df.loc[loc, ['mean_b']] = mean[2]
        df.loc[loc, ['std_r']] = std[0]
        df.loc[loc, ['std_g']] = std[1]
        df.loc[loc, ['std_b']] = std[2]
        df.loc[loc, ['transparency']] = cloud.texture.transparency()
        df.loc[loc, ['mean_diff_edges_r']] = mean_diff_edges[0]
        df.loc[loc, ['mean_diff_edges_g']] = mean_diff_edges[1]
        df.loc[loc, ['mean_diff_edges_b']] = mean_diff_edges[2]

df.to_csv(f'{datetime.today().strftime("%Y-%m-%d-%H-%M-%S")},'
          f'num_img-{num_img},num_clouds-{num_clouds}')
