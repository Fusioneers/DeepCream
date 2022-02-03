import os

import cv2 as cv

import cloud_analysis.analysis

path = os.path.normpath(os.path.join(os.getcwd(), os.path.normpath(
    'DeepCream/sample_data/Data/zz_astropi_1_photo_364.jpg')))
print(path)


def test_is_not_none():
    img = cv.imread(path)
    assert img is not None

    analysis = cloud_analysis.analysis.Analysis(img, 5, 0.1)
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
    assert all([cloud.mean() is not None for cloud in analysis.clouds])
    assert all([cloud.std() is not None for cloud in analysis.clouds])
    assert all([cloud.transparency() is not None for cloud in analysis.clouds])
    assert all([cloud.edges(50, 50) is not None for cloud in analysis.clouds])
    assert all([cloud.mean_diff_edges(50, 50) is not None for cloud in
                analysis.clouds])
