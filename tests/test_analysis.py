import os
from tests.__init__ import rep_path
import sys

# sys.path.append(os.path.join(rep_path, 'DeepCream'))
# print(rep_path)
import cv2 as cv

from DeepCream.cloud_analysis.analysis import Analysis

# ----- VERY FRAGILE -----
path = os.path.normpath(
    os.path.join(rep_path, 'sample_data/Data/zz_astropi_1_photo_364.jpg'))


# TODO multiple functions
def test_is_not_none():
    img = cv.imread(path)
    assert img is not None

    analysis = Analysis(img, 2000, 20, 100)
    assert analysis is not None
    assert analysis.clouds is not None
    assert analysis.orig is not None
    assert analysis.mask is not None
    assert all([cloud is not None for cloud in analysis.clouds])

    assert all([cloud.contour is not None for cloud in analysis.clouds])
    assert all(
        [cloud.contour_perimeter is not None for cloud in analysis.clouds])
    assert all([cloud.contour_area is not None for cloud in analysis.clouds])
    assert all([cloud.hull is not None for cloud in analysis.clouds])
    assert all([cloud.hull_perimeter is not None for cloud in analysis.clouds])
    assert all([cloud.hull_area is not None for cloud in analysis.clouds])

    assert all([cloud.roundness() is not None for cloud in analysis.clouds])
    assert all([cloud.convexity() is not None for cloud in analysis.clouds])
    assert all([cloud.compactness() is not None for cloud in analysis.clouds])
    assert all([cloud.solidity() is not None for cloud in analysis.clouds])
    assert all(
        [cloud.rectangularity() is not None for cloud in analysis.clouds])
    assert all([cloud.elongation() is not None for cloud in analysis.clouds])
    assert all([cloud.mean() for cloud in analysis.clouds])
    assert all([cloud.std() for cloud in analysis.clouds])
    assert all([cloud.transparency() is not None for cloud in analysis.clouds])
    assert all([cloud.edges(50, 50).size for cloud in analysis.clouds])
    assert all([cloud.diff_edges(20, 20) is not None for cloud in
                analysis.clouds])


if __name__ == '__main__':
    test_is_not_none()
