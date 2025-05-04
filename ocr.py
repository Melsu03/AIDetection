import cv2
import pytesseract
import numpy as np
import argparse
from pytesseract import Output

class ImageTextExtractor:
    def __init__(self, image_array):
        self.img_source = image_array
        if self.img_source is None:
            raise ValueError(f"Error: Unable to access image data")

    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 128, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    def denoise(self, image):
        return cv2.medianBlur(image, 5)  # Adjust kernel size as needed
        
    def detect_document_edges(self, image):
        """Detect the edges of a document in the image and return the corners"""
        # Convert to grayscale and blur
        gray = self.get_grayscale(image)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blur, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find the largest contour that could be a document
        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # If the contour has 4 points, it's likely a document
            if len(approx) == 4:
                return approx
        
        # If no suitable contour is found, return None
        return None
    
    def order_points(self, pts):
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
        # Convert to numpy array if not already
        pts = np.array(pts).reshape(4, 2)
        
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left point has the smallest sum of coordinates
        # Bottom-right point has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has the smallest difference of coordinates
        # Bottom-left point has the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def four_point_transform(self, image, pts):
        """Apply perspective transform to get a top-down view of the document"""
        # Get ordered points
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Compute height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Create destination points for transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def preprocess_document(self, image):
        """Detect document edges and apply perspective correction"""
        # Make a copy of the image to avoid modifying the original
        processed_img = image.copy()
        
        # Detect document edges
        corners = self.detect_document_edges(processed_img)
        
        # If document edges are detected, apply perspective transform
        if corners is not None:
            # Reshape corners to the format expected by four_point_transform
            corners = corners.reshape(4, 2)
            processed_img = self.four_point_transform(processed_img, corners)
            
            # Apply additional preprocessing for better OCR
            gray = self.get_grayscale(processed_img)
            denoised = self.denoise(gray)
            return denoised
        
        # If no document edges detected, just apply basic preprocessing
        gray = self.get_grayscale(processed_img)
        denoised = self.denoise(gray)
        return denoised

    def extract_text(self):
        # Preprocess the document image
        processed_img = self.preprocess_document(self.img_source)
        
        # Apply OCR on the processed image
        data = pytesseract.image_to_data(processed_img, output_type=Output.DICT)
        n_boxes = len(data['text'])

        # Organize text into paragraphs
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

        # Build the paragraphs as one whole blob with spaced paragraphs
        result_text = ""
        for block_num in paragraphs:
            for par_num in paragraphs[block_num]:
                paragraph_text = ' '.join(paragraphs[block_num][par_num])
                result_text += paragraph_text + "\n\n"  # Add a newline to separate paragraphs

        return result_text
