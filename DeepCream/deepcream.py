import cv2
from DeepCream.cloud_detection.cloud_filter import CloudFilter
from DeepCream.constants import (ABS_PATH)


def start():
    cf = CloudFilter(ABS_PATH + "/data/input/")
    mask = cf.evaluate_image("photo_00150_51846468570_o.jpg")
    cv2.imwrite(ABS_PATH + "/data/output/test.TIF", mask)
    return "200 Successful"
