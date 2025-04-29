import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from picamera2 import Picamera2, controls
from picamera2.previews.qt import QGlPicamera2
import cv2
import numpy as np
from ui_main_window3 import QtMainWindow
from ocr import ImageTextExtractor  # Import the ImageTextExtractor class
from infer_model2 import AIPlagiarismDetector  # Import the AITextDetector class
from flask import Flask, jsonify
from threading import Thread
from queue import Queue
from threading import Lock

# Initialize Flask app
app = Flask(__name__)

# Shared variable to store the result
shared_result = { "status": "", "result": "", "perplexity": "", "burstiness": "", "intrp": "" }

@app.route('/result', methods=['GET'])
def get_result():
    return jsonify(shared_result)

class CaptureThread(QThread):
    capture_done = pyqtSignal(np.ndarray)

    def __init__(self, picam2):
        super().__init__()
        self.picam2 = picam2

    def run(self):
        # Capture the image directly into a NumPy array
        image_array = self.picam2.capture_array()
        self.capture_done.emit(image_array)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.ui = QtMainWindow()
        self.ui.setupUi(self)

        # Initialize Picamera2
        self.picam2 = Picamera2()

        # Set up camera preview configuration
        preview_width = 1080 
        preview_height = int(self.picam2.sensor_resolution[1] * preview_width / self.picam2.sensor_resolution[0])
        preview_config = self.picam2.create_preview_configuration(main={"size": (preview_width, preview_height)})
        self.picam2.configure(preview_config)
        
        # Replace wgtCamera placeholder with QGlPicamera2 to display the camera feed
        self.camera_widget = QGlPicamera2(self.picam2, width=preview_width, height=preview_height, keep_ar=True)
        
        # Add camera_widget to the layout in place of the placeholder widget
        self.ui.verticalLayout.replaceWidget(self.ui.wgtCamera, self.camera_widget)
        self.ui.wgtCamera.deleteLater()  # Remove the original placeholder widget

        # Start the camera preview
        self.picam2.start()
        #self.picam2.set_controls({"AfMode": 1})

        # Connect capture button to the capture function
        self.ui.btnSingleCap.clicked.connect(self.capture_image)
        
        # Connect btnFromFile to the load_file method
        self.ui.btnFromFile.clicked.connect(self.load_file)

        # Add batch processing components
        self.image_queue = Queue()
        self.queue_lock = Lock()
        self.batch_mode = False
        
        # Connect batch processing buttons that are already in the UI
        self.ui.btnBatchCap.clicked.connect(self.toggle_batch_mode)
        self.ui.btnBatchTake.clicked.connect(self.capture_image)
        self.ui.btnBatchNext.clicked.connect(self.process_batch)
        self.ui.btnBatchStop.clicked.connect(self.stop_batch_mode)

        # Initially hide batch operation buttons
        self.ui.btnBatchTake.hide()
        self.ui.btnBatchNext.hide()
        self.ui.btnBatchStop.hide()
        
        # Initialize the AITextDetector
        self.detector = AIPlagiarismDetector('model/trained_model2.pkl')

    def toggle_batch_mode(self):
        """Toggle between single capture and batch capture modes"""
        self.batch_mode = True
        
        # Show batch operation buttons, hide single capture button
        self.ui.btnSingleCap.hide()
        self.ui.btnBatchCap.hide()
        self.ui.btnBatchTake.show()
        self.ui.btnBatchNext.show()
        self.ui.btnBatchStop.show()
        
        # Clear any existing queue
        with self.queue_lock:
            while not self.image_queue.empty():
                self.image_queue.get()
        
        print("Batch mode activated. Capture images and then process them.")
    
    def process_batch(self):
        """Process all images in the queue"""
        global shared_result
        
        with self.queue_lock:
            queue_size = self.image_queue.qsize()
            
            if queue_size == 0:
                print("No images in batch to process")
                shared_result = {
                    "status": "error", 
                    "message": "No images in batch to process"
                }
                return
            
            print(f"Processing {queue_size} images in batch")
            batch_results = []
            
            while not self.image_queue.empty():
                image_array = self.image_queue.get()
                
                # Process the image
                extractor = ImageTextExtractor(image_array)
                extracted_text = extractor.extract_text()
                result = self.detector.detect_ai_text(extracted_text)
                
                batch_results.append({
                    "text": extracted_text,
                    "result": result[0],
                    "perplexity": result[1],
                    "burstiness": result[2],
                    "interpretation": result[3]
                })
        
        # Calculate summary statistics
        ai_count = sum(1 for r in batch_results if "AI-generated" in r["result"])
        human_count = len(batch_results) - ai_count
        
        # Update shared result with batch summary
        shared_result = {
            "status": "batch_complete",
            "total_images": len(batch_results),
            "ai_detected": ai_count,
            "human_detected": human_count,
            "details": batch_results
        }
        
        print(f"Batch processing complete. AI detected in {ai_count}/{len(batch_results)} images")
        
        # Return to normal mode after processing
        self.stop_batch_mode()
    
    def stop_batch_mode(self):
        """Exit batch mode and return to normal operation"""
        self.batch_mode = False
        
        # Show normal buttons, hide batch operation buttons
        self.ui.btnCapture.show()
        self.ui.btnBatchCap.show()
        self.ui.btnBatchTake.hide()
        self.ui.btnBatchNext.hide()
        self.ui.btnBatchStop.hide()
        
        print("Batch mode deactivated.")

    def load_file(self):
        global shared_result

        # Get the file path from the lineEdit widget
        file_path = self.ui.lineEdit.text()

        try:
            # Load the image using OpenCV
            image_array = cv2.imread(file_path)

            if image_array is None:
                raise FileNotFoundError("Could not load the image from the provided path.")

            # Process the loaded image
            extractor = ImageTextExtractor(image_array)
            extracted_text = extractor.extract_text()
            print("Extracted Text from File:")
            print(extracted_text)
            
            # Detect AI-generated text
            result = self.detector.detect_ai_text(extracted_text)
            print(result)

            # Update the shared result
            shared_result = {"status": "success", "result": result[0], "perplexity": result[1], "burstiness": result[2], "intrp": result[3]}
        except Exception as e:
            print(f"Error loading file: {e}")
            shared_result = {"status": "error", "result": str(e)}

    def capture_image(self):
        # Create and start the capture thread
        self.capture_thread = CaptureThread(self.picam2)
        self.capture_thread.capture_done.connect(self.on_capture_done)
        self.capture_thread.start()

    # def on_capture_done(self, image_array):
    #     global shared_result

    #     # Use ImageTextExtractor to extract text from the image array
    #     extractor = ImageTextExtractor(image_array)
    #     extracted_text = extractor.extract_text()
    #     print("Extracted Text:")
    #     print(extracted_text)
        
    #     # Use AITextDetector to check if the extracted text is AI-generated
    #     result = self.detector.detect_ai_text(extracted_text)
    #     print(result)

    #     # Update the shared result
    #     shared_result = {"status": "success", "result": result[0], "perplexity": result[1], "burstiness": result[2], "intrp": result[3]}
    
    def on_capture_done(self, image_array):
        """Handle captured image based on current mode"""
        global shared_result
        
        if self.batch_mode:
            # In batch mode, add to queue instead of processing immediately
            with self.queue_lock:
                self.image_queue.put(image_array)
                queue_size = self.image_queue.qsize()
                
            print(f"Added image to batch. Queue size: {queue_size}")
            shared_result = {
                "status": "batch_update", 
                "message": f"Image added to batch. Total: {queue_size}"
            }
        else:
            # In single capture mode, process immediately as before
            extractor = ImageTextExtractor(image_array)
            extracted_text = extractor.extract_text()
            print("Extracted Text:")
            print(extracted_text)
            
            # Use AITextDetector to check if the extracted text is AI-generated
            result = self.detector.detect_ai_text(extracted_text)
            print(result)

            # Update the shared result
            shared_result = {
                "status": "success", 
                "result": result[0], 
                "perplexity": result[1], 
                "burstiness": result[2], 
                "intrp": result[3]
            }

def run_flask():
    # Start Flask app on a separate thread
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
