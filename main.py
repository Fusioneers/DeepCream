import cv2
from cloud_detection.cloud_filter import CloudFilter

cf = CloudFilter(weight_ai=0.5, image_directory='sample_data/Data/', tpu_support=True)

clouds, mask = cf.evaluate_image('zz_astropi_1_photo_215.jpg')

while True:
    cv2.imshow('Clouds', clouds)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break