import time

import cv2 as cv
import matplotlib.pyplot as plt

from analysis import Analysis


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    plt.show()


image = cv.imread('../sample_data/Data/zz_astropi_1_photo_364.jpg')
analysis = Analysis(image, 5, 0.1)

print(analysis.clouds[1].convexity())

# dtime = time.time()
# for cloud in analysis.clouds:
#     try:
#         print(cloud.mean_diff_edges(3, 50, 500))
#     except Exception:
#         print(Exception)
#
# print(f'time: {time.time() - dtime}')
