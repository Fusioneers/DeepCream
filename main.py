import cv2

from cloud_detection.cloud_filter import CloudFilter

cf = CloudFilter(weight_ai=0.8)

clouds, mask = cf.evaluate_image('./sample_data/image1.jpg')

while True:
    cv2.imshow('Clouds', clouds)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
