import pytest
import os
import cv2 as cv
import numpy as np
from PIL import Image

from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_filter import CloudFilter

from DeepCream.constants import ABS_PATH

path = os.path.normpath(
    os.path.join(ABS_PATH, 'data/Data/zz_astropi_1_photo_364.jpg'))

# TODO test whether the values are meaningful
# TODO test the performance

cloud_filter = CloudFilter()
mask, _ = cloud_filter.evaluate_image(Image.open(path))

out = cv.resize(mask, (mask.shape[1], mask.shape[0]))


def test_create_analysis():
    img = cv.imread(path)
    assert img is not None

    analysis = Analysis(img, mask, 5, 0.2)
    assert analysis is not None
    assert analysis.clouds


analysis = Analysis(cv.imread(path), mask, 5, 0.2)


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
