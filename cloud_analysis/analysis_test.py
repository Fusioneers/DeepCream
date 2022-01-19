from analysis import Analysis
import os
import cv2 as cv

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis_test.py')

analysis = Analysis(cv.imread(path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'), 5)

print(analysis.clouds[-1].edges(0.01, 3, 100))
