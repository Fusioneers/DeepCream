import pandas as pd
import os
import cv2 as cv
from tqdm import tqdm
from analysis import Analysis

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\save_analysis_data.py')

# TODO get data for a lot of images and compare features for hopefully meaningful results


directory = path + r'sample_data\set_1'
num_clouds = 5


num_img = len(os.listdir(directory))

columns = ['roundness',
           'rectangularity',
           'convexity',
           'compactness',
           'solidity',
           'elongation',
           'mean_r',
           'mean_g',
           'mean_b',
           'std_r',
           'std_g',
           'std_b',
           'contrast',
           'glds_skewness_0.5',
           'transparency',
           'edge_width', ]
df = pd.DataFrame(columns=columns)

for i, img in tqdm(enumerate(os.scandir(directory)), total=num_img):
    pass
    analysis = Analysis(cv.imread(img.path), num_clouds)

    for j, cloud in enumerate(analysis.clouds):
        std = cloud.texture.std()
        mean = cloud.texture.mean()

        df.loc[num_clouds * i + j, ['roundness']] = cloud.shape.roundness()
        df.loc[num_clouds * i + j, ['rectangularity']] = cloud.shape.rectangularity()
        df.loc[num_clouds * i + j, ['convexity']] = cloud.shape.convexity()
        df.loc[num_clouds * i + j, ['compactness']] = cloud.shape.compactness()
        df.loc[num_clouds * i + j, ['solidity']] = cloud.shape.solidity()
        df.loc[num_clouds * i + j, ['elongation']] = cloud.shape.elongation()
        df.loc[num_clouds * i + j, ['mean_r']] = mean[0]
        df.loc[num_clouds * i + j, ['mean_g']] = mean[1]
        df.loc[num_clouds * i + j, ['mean_b']] = mean[2]
        df.loc[num_clouds * i + j, ['std_r']] = std[0]
        df.loc[num_clouds * i + j, ['std_g']] = std[1]
        df.loc[num_clouds * i + j, ['std_b']] = std[2]
        df.loc[num_clouds * i + j, ['contrast']] = cloud.texture.contrast()
        df.loc[num_clouds * i + j, ['glds_skewness_0.5']] = cloud.texture.glds_skewness(0.5)
        df.loc[num_clouds * i + j, ['transparency']] = cloud.texture.transparency()
        df.loc[num_clouds * i + j, ['edge_width']] = cloud.edge_width()

df.to_csv(f'num_img-{num_img},num_clouds-{num_clouds}')
