from analysis import Analysis
import os
import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    plt.show()


path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis_test.py')

analysis = Analysis(cv.imread(path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'), 5)

plot(analysis.clouds[-1].img)
dtime = time.time()
edges = analysis.clouds[-1].edges(0.3, 300, 300)
print(time.time() - dtime)
plot(edges)
avr = np.tile(np.mean(edges, axis=0), (edges.shape[1], 1, 1)).astype('int')
print(avr.shape)
print(avr)
plt.imshow(avr)
plt.show()
