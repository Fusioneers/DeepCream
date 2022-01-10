import cv2 as cv
import numpy as np
import os
from CloudDetection.CloudFilter import CloudFilter

path = os.path.realpath(__file__).removesuffix(r'\CloudAnalysis\CloudAnalysis.py')

cf = CloudFilter()

cf.load_image('./test-data/image0.jpg')

cf.evaluate_image('./processed-data/')


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

def rescale_image(lines, orig):


def get_clouds(img, min_area):
    img = cv.medianBlur(img, 3)
    contours, _ = cv.findContours(img.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area:
            new_contours.append(cnt)
