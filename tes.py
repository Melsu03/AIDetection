import cv2
import pytesseract
import numpy as np
from pytesseract import Output
#from picamera2 import Picamera2, Preview

img_source = cv2.imread('sample/sample7.jpeg')

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 128, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def denoise(image):
   return cv2.medianBlur(image, 5)  # Adjust kernel size as needed

gray = get_grayscale(img_source)

data = pytesseract.image_to_data(gray, output_type=Output.DICT)
n_boxes = len(data['text'])

paragraphs = {}
for i in range(n_boxes):
    block_num = data['block_num'][i]
    par_num = data['par_num'][i]
    text = data['text'][i]
    if text.strip():
        if block_num not in paragraphs:
            paragraphs[block_num] = {}
        if par_num not in paragraphs[block_num]:
            paragraphs[block_num][par_num] = []
        paragraphs[block_num][par_num].append(text)

# Print the paragraphs as one whole blob with spaced paragraphs
for block_num in paragraphs:
    for par_num in paragraphs[block_num]:
        paragraph_text = ' '.join(paragraphs[block_num][par_num])
        print(paragraph_text)
        print("\n")  # Add a newline to separate paragraphs
