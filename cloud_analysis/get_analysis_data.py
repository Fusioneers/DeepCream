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
    # img = cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
    # cv.imshow('', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    plt.imshow(img, cmap='gray')
    plt.show()


def contrast_img(image, n=2):
    out = np.zeros(image.shape)

    for offset_x in range(-n, n + 1):
        for offset_y in range(-n, n + 1):
            out += np.abs(image - np.roll(image, (offset_x, offset_y), axis=(1, 0)))

    return np.floor(np.divide(out, (2 * n + 1) ** 2))


dtime = time.time()

analysis = Analysis(cv.imread(img_path), num_clouds, distance, num_angles)
print('finished constructing')

plot(analysis.clouds[-1].img)
contrast_a = contrast_img(analysis.clouds[-1].texture.grey, 10)
plot(contrast_a)
contrast_b = contrast_img(contrast_a, 10)
plot(contrast_b)
contrast_c = contrast_img(contrast_b, 10)
plot(contrast_c)
contrast_d = contrast_img(contrast_c, 10)
plot(contrast_d)
contrast_e = contrast_img(contrast_d, 10)
plot(contrast_e)
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

# time = time.time()
# img = analysis.clouds[0].texture.contrast_img()
# print(time.time() - dtime)
# plot(img)
