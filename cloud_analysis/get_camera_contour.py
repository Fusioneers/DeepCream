import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = 'C:\\Users\\Daniel Meiborg\\OneDrive - birklehof.de\\Dokumente\\PycharmProjects\\Fusioneers\\DeepCream\\' \
       'sample_data\\visible_area_mask.png'
img = cv.cvtColor(cv.imread(path), cv.COLOR_RGB2GRAY)
print(img.shape)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print(contours)
black = np.zeros(img.shape)
mask = cv.drawContours(black, contours, -1, (255, 255, 255), -1)
plt.imshow(mask)
plt.show()

joined_contours = []
for arr in contours:
    joined_contours += arr.tolist()
joined_contours = np.array(joined_contours)
joined_contours = joined_contours.squeeze()
print(joined_contours)
np.save('camera_contour', joined_contours, joined_contours)
