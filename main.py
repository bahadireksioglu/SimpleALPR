from carclassification import carclassification 
from platerecognition import licenseplatedetection, textdetection
from otherprocesses import servo, ultrasonicdistancesensor
import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    # by frame
    car_class_name = ""
    detected_text = ""

    distance = ultrasonicdistancesensor.ultrasonic_distance_sensor()
    if distance < 10:
        ret, frame = cap.read()
        img, car_class_name = carclassification.find_objects(img)

        if car_class_name == "CAR" or car_class_name == "TRUCK" or car_class_name == "BUS":
            img = licenseplatedetection.detect_license_plate(img)
            detected_text = textdetection.text_detection(img)

            if detected_text != "":
                servo.servo_forward_ninety_degree()
                time.sleep(10)
                servo.servo_backward_ninety_degree()

                print(car_class_name, detected_text)

        