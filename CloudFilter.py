import datetime
import cv2
import numpy as np
import numpy


class CloudFilter:
    def __init__(self, night_threshold=20, cloudless_threshold=0.02, h_min=0, h_max=179, s_min=0, s_max=50, v_min=145,
                 v_max=255, contrast=1, brightness=0, blur=3, canny_threshold_1=20, canny_threshold_2=150):

        # Thresholds for stopping the image processing
        self.nightThreshold = night_threshold
        self.cloudlessThreshold = cloudless_threshold

        # Set the contrast, brightness and blur which should be applied to the image
        self.contrast = contrast
        self.brightness = brightness
        self.blur = blur

        # Set the min/max Hue, Saturation, Value values for the image filtering
        self.hMin = h_min
        self.hMax = h_max
        self.sMin = s_min
        self.sMax = s_max
        self.vMin = v_min
        self.vMax = v_max

        # Set the thresholds for the cv.Canny(...) function
        self.cannyThreshold1 = canny_threshold_1
        self.cannyThreshold2 = canny_threshold_2

        # Define variables which are set later
        self.original = None
        self.gray = None

    def load_image(self, image_path):
        # Load image
        self.original = cv2.imread(image_path)

        if self.original is None:
            return "Error loading the image"
        else:
            # Compute the grayscale image
            self.gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
            return "Image successfully loaded"

    def evaluate_image(self, output_path):
        # Check if image is loaded
        if self.original is None or self.gray is None:
            return "No image loaded"

        # Calculate the average color of the grayscale image
        mean = self.gray.mean()

        # Check if the image is overall to dark
        if mean < self.nightThreshold:
            return "Too dark"

        multi_color_original = self.original

        # Apply the blur to the image
        if self.blur > 1:
            multi_color_blur = cv2.blur(multi_color_original, (self.blur, self.blur))
        else:
            multi_color_blur = multi_color_original

        # Apply changes in contrast and brightness
        multi_color_image = cv2.convertScaleAbs(multi_color_blur, alpha=self.contrast, beta=self.brightness)

        # Set lower and upper boundaries to filter out
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])

        # Create the image mask to filter out everything but the clouds
        hsv = cv2.cvtColor(multi_color_image, cv2.COLOR_BGR2HSV)
        multi_color_mask = cv2.inRange(hsv, lower, upper)

        # Compute the cloud coverage
        cloud_share = numpy.divide(cv2.countNonZero(multi_color_mask), self.original.size)
        # print(str(cloud_share) + ' cloud coverage')

        # Check if the cloud coverage is too low
        if cloud_share < self.cloudlessThreshold:
            return "Too cloudless"

        # Apply the mask to filter out everything but the clouds
        multi_color_output = cv2.bitwise_and(multi_color_original, multi_color_original, mask=multi_color_mask)

        # Compute the outline of the clouds
        multi_color_lines = cv2.Canny(cv2.bitwise_and(multi_color_image, multi_color_image, mask=multi_color_mask),
                                      self.cannyThreshold1, self.cannyThreshold2, None, 3, True)

        multi_color_control_image = cv2.bitwise_not(multi_color_image, multi_color_image, mask=multi_color_lines)

        # Save the output images to the supplied path

        # The output only containing the clouds
        cv2.imwrite(output_path + str(datetime.datetime.now().date()) + '-color-output.jpg',
                    multi_color_output)

        # The outline of the clouds
        cv2.imwrite(output_path + str(datetime.datetime.now().date()) + '-lines-output.jpg', multi_color_lines)

        # The modified image with an overlay of the cloud outline
        cv2.imwrite(output_path + str(datetime.datetime.now().date()) + '-control-output.jpg',
                    multi_color_control_image)

        return "Process ended successfully (images were saved)"

        # maybe TODO create mask from blue channel to best filter out water
