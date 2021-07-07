import RPi.GPIO as GPIO
import time


GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.OUT)
servo = GPIO.PWM(11,50)

servo.start(0)
time.sleep(2) # OR YOUR CHOICE

def servo_forward_ninety_degree(servo):
    servo.ChangeDutyCycle(7)
    time.sleep(2)

def servo_backward_ninety_degree(servo):
    servo.ChangeDutyCycle(2)
    time.sleep(1)
    servo.ChangeDutyCycle(0)


servo.stop()
GPIO.cleanup()

