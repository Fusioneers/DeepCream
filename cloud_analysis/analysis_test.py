import cv2 as cv

from analysis import Analysis

image = cv.imread('../sample_data/Data/zz_astropi_1_photo_364.jpg')


def test_analysis_init():
    analysis = Analysis(image, 5, 0.1)
    assert analysis is not None

# dtime = time.time()
# for cloud in analysis.clouds:
#     try:
#         print(cloud.mean_diff_edges(3, 50, 500))
#     except Exception:
#         print(Exception)
#
# print(f'time: {time.time() - dtime}')
