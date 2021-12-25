import cv2
import numpy as np
from numpy import asarray
from PIL import Image
from cloud_detection.unet_model import unet_model


class CloudFilter:
    def __init__(self, night_threshold=20, cloudless_threshold=0.02, binary_cloud_threshold=100, h_min=0, h_max=179, s_min=0, s_max=50, v_min=145,
                 v_max=255, contrast=1, brightness=0, blur=3, weight_ai=0.7):

        # Thresholds for stopping the image processing
        self.nightThreshold = night_threshold
        self.cloudlessThreshold = cloudless_threshold
        self.binaryCloudThreshold = binary_cloud_threshold

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

        # The weight of the ai prediction
        self.weightAi = weight_ai

        # Load the machine learning model
        self.HEIGHT = 96 * 2
        self.WIDTH = 128 * 2
        self.CHANNELS = 3
        self.model = unet_model(self.HEIGHT, self.WIDTH, self.CHANNELS)
        self.model.load_weights('./cloud_detection/clouds_test.hdf5')

    def load_image(self, image_path):
        normal = cv2.resize(cv2.imread(image_path), (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
        scaled = Image.open(image_path)

        if normal is None or scaled is None:
            return None, None, None
        else:
            scaled.thumbnail((self.WIDTH, self.HEIGHT))
            scaled = asarray(scaled)
            scaled = scaled.astype('float32')
            scaled /= 255.0

            gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)

        return normal, scaled, gray

    def ai_generate_image_mask(self, original):
        # Let the ai predict where the clouds are
        pred = (self.model.predict(np.array([original])).reshape(self.HEIGHT, self.WIDTH, 1))
        
        # Convert the prediction to a mask
        mask = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return mask

    def cv2_generate_image_mask(self, original):
        # Blur the image
        if self.blur > 1:
            blurred = cv2.blur(original, (self.blur, self.blur))
        else:
            blurred = original

        # Adjust contrast and brightness
        adjusted = cv2.convertScaleAbs(blurred, alpha=self.contrast, beta=self.brightness)

        # Set lower and upper boundaries to filter out
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])

        # Create the image mask to filter out everything but the clouds
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        rough_mask = cv2.inRange(hsv, lower, upper)
        hsv = cv2.bitwise_and(hsv, hsv, mask=rough_mask)

        # Create an empty mask with the right dimensions
        fine_mask = np.zeros(hsv.shape, hsv.dtype)

        # Calculate for each pixel how likely it is to be a cloud and five it back as a mask
        for y in range(hsv.shape[0]):
            for x in range(hsv.shape[1]):
                # TODO RuntimeWarning: overflow encountered in ubyte_scalars ??
                hue = (hsv[y, x, 0] > 0) * np.clip(255 - (hsv[:, :, 0].max()-hsv[y, x, 2]), 0, 255)

                saturation = (hsv[y, x, 1] > 0) * np.clip(255 - (hsv[y, x, 1]), 0, 255)

                value = np.clip(255 + hsv[y, x, 2] - self.vMax, 0, 255)

                fine_mask[y, x] = [hue, saturation, value]

        return cv2.cvtColor(fine_mask, cv2.COLOR_BGR2GRAY)

    def evaluate_image(self, image_path, dark="Too dark"):
        normal, scaled, gray = self.load_image(image_path)

        if normal is None or scaled is None or gray is None:
            return None, None

        # Check if the image is overall to dark
        if gray.mean() < self.nightThreshold:
            return dark

        # Compute the two masks
        ai_mask = self.ai_generate_image_mask(scaled)
        hsv_mask = self.cv2_generate_image_mask(normal)

        # Combine the two masks according
        mask = cv2.addWeighted(ai_mask, self.weightAi, hsv_mask, (1-self.weightAi), 0.0)

        # Normalize the resulting maks and make it binary
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ret, mask = cv2.threshold(mask, self.binaryCloudThreshold, 255, cv2.THRESH_BINARY)

        # Apply the mask to filter out everything but the clouds
        multi_color_output = cv2.bitwise_and(normal, normal, mask=mask)

        return mask, multi_color_output
