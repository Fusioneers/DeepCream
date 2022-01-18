import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import seaborn as sns
import os
import time
import cv2 as cv
from cloud_analysis import Analysis

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\get_analysis_data.py')

# TODO get data for a lot of images and compare features for hopefully meaningful results


img_path = path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'
num_clouds = 5
distance = 20
num_angles = 16
img = cv.imread(img_path)


def plot(img):
    plt.imshow(img, cmap='gray')
    plt.show()


dtime = time.time()

analysis = Analysis(cv.imread(img_path), num_clouds, distance, num_angles)
print('finished constructing')

plot(analysis.clouds[-1].img)
plot(analysis.clouds[-1].texture.contrast_img())
print(time.time() - dtime)

# print('\n################################\n')
# for cloud in analysis.clouds:
# print(cloud.shape)
# print('\n--------------------------------\n')
# print(cloud.texture)
# print('\n################################\n')
# contrast_img = cloud.texture.double_contrast_img()
# print(contrast_img)
# print(np.max(contrast_img))
# print(contrast_img.shape)
# plot(contrast_img)

# for channel in range(3):
#    data = analysis.clouds[0].texture.dis()[channel]
#    print(data.shape)
#    df = pd.DataFrame({'value': data})
#    print(df)
#    print(df.describe())
#    sns.displot(df, kde=True, bins=range(1, 256))
#    plt.show()
#