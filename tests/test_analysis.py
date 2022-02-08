import pytest
import os
import cv2 as cv
import numpy as np

from DeepCream.constants import ABS_PATH

from DeepCream.cloud_analysis.analysis import Analysis

path = os.path.normpath(
    os.path.join(ABS_PATH, 'data/Data/zz_astropi_1_photo_364.jpg'))

# TODO find better solution, but has low priority
analysis = 0


# TODO test whether the values are meaningful
# TODO test the performance

def test_create_analysis():
    global analysis

    img = cv.imread(path)
    assert img is not None

    # TODO correct parameters
    analysis_ = Analysis(img, 5, 5, 20)
    assert analysis_ is not None
    assert analysis_.clouds
    analysis = analysis_


@pytest.mark.parametrize('obj', [
    'orig',
    'mask',
])
def test_is_not_none(obj):
    assert getattr(analysis, obj).size


@pytest.mark.parametrize('obj', [
    'contour',
    'hull',
])
def test_is_not_empty(obj):
    assert np.array(getattr(analysis.clouds[0], obj)).size


@pytest.mark.parametrize('obj', [
    'contour_perimeter',
    'contour_area',
    'hull_perimeter',
    'hull_area',
])
def test_not_none_cloud_attributes(obj):
    assert getattr(analysis.clouds[0], obj)


@pytest.mark.parametrize('obj', [
    'roundness',
    'convexity',
    'compactness',
    'solidity',
    'rectangularity',
    'elongation',
    'transparency',
    'mean',
    'std',
])
def test_not_none_methods(obj):
    assert getattr(analysis.clouds[0], obj)()
