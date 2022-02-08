import cv2 as cv
import matplotlib.pyplot as plt

from DeepCream.cloud_analysis.analysis import Analysis

# TODO __init__ has changed, see in the docstring of analysis
analysis = Analysis(
    cv.imread('../../data/Data/zz_astropi_1_photo_364.jpg'), 5)

# Load the image
img = cv.imread("shapes/ellipse.png")
# Convert to greyscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Convert to binary image by thresholding
_, threshold = cv.threshold(img_gray, 245, 255, cv.THRESH_BINARY_INV)
# Find the contours
contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# For each contour approximate the curve and detect the shapes.
for cnt in contours:
    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    cv.drawContours(img, [approx], 0, 0, 3)
    # Position for writing text
    x, y = approx[0][0]

    if len(approx) == 3:
        cv.putText(img, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif len(approx) == 4:
        cv.putText(img, "Rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif len(approx) == 5:
        cv.putText(img, "Pentagon", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif 6 < len(approx) < 15:
        cv.putText(img, "Ellipse", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    else:
        cv.putText(img, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
cv.imshow("final", img)
cv.waitKey(0)

plt.plot(img)

plt.show()
