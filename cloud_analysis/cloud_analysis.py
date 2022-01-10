import cv2 as cv
import numpy as np
import os
from cloud_detection.cloud_filter import CloudFilter

path = os.path.realpath(__file__).removesuffix(r'\CloudAnalysis\CloudAnalysis.py')


class Cloud:

    def __init__(self, img):
        self.img = img
        self.size = cv.countNonZero(img)

    def get_form(self):
        pass

    def get_texture(self):
        # texture
        # patterns
        pass

    def get_transparency(self):
        pass

    def get_edges(self):
        pass

    def analyze_cloud(self):
        print(self.size)


def rescale_image(mask, orig):
    width, height, channels = orig.shape
    lines_re = cv.resize(mask, (width, height))
    return cv.bitwise_and(mask, orig)


def get_clouds(img, min_area):
    img = cv.medianBlur(img, 3)
    contours, _ = cv.findContours(img.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area:
            new_contours.append(cnt)
    cv.imshow('', cv.drawContours(img, new_contours, -1, (255, 255, 255), -1))


if __name__ == '__main__':
    cf = CloudFilter()
    cf.load_image(path + '/sample_data/Data/zz_astropi_1_photo_364.jpg')
    mask, color_image = cf.evaluate_image(path + '/cloud_analysis/')