from ./analysis import Analysis
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis_test.py')

analysis = Analysis(cv.imread(path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'), 5)



# Load the image
img = cv2.imread("shapes/ellipse.png")
# Convert to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convert to binary image by thresholding
_, threshold = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY_INV)
# Find the contours
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# For each contour approximate the curve and
# detect the shapes.
for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approx], 0, (0), 3)
    # Position for writing text
    x, y = approx[0][0]

    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    elif 6 < len(approx) < 15:
        cv2.putText(img, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
cv2.imshow("final", img)
cv2.waitKey(0)

plt.plot(img)

plt.show()