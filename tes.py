import sys
import argparse
import cv2
import pytesseract
import numpy as np
from pytesseract import Output

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

# Set up argument parser
parser = argparse.ArgumentParser(description='Process an image to extract text.')
parser.add_argument('filename', type=str, help='The path to the image file')
args = parser.parse_args()

img_source = cv2.imread(args.filename)

if img_source is None:
    print(f"Error: Unable to open image file {args.filename}")
    sys.exit(1)

gray = get_grayscale(img_source)

data = pytesseract.image_to_data(gray, output_type=Output.DICT)
n_boxes = len(data['text'])

# Draw bounding boxes and text on the image
for i in range(n_boxes):
    if int(data['conf'][i]) > 30:  # Filter out weak confidence text
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        text = data['text'][i]
        if text.strip():
            img_source = cv2.rectangle(img_source, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_source = cv2.putText(img_source, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', img_source)
cv2.waitKey(0)
cv2.destroyAllWindows()

paragraphs = {}
for i in range(n_boxes):
    if int(data['conf'][i]) > 30:  # Filter out weak confidence text
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