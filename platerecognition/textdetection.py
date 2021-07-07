import cv2
import pytesseract

# main setup 

pytesseract.pytesseract.tesseract_cmd = 'YOUR TESSERACT.EXE PATH'

def text_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_text = pytesseract.image_to_string(img)
    return detected_text

