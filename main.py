import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from picamera2 import Picamera2, controls
from picamera2.previews.qt import QGlPicamera2
import cv2
import numpy as np
from ui_main_window import QtMainWindow
from ocr import ImageTextExtractor  # Import the ImageTextExtractor class
from infer_model import AIPlagiarismDetector  # Import the AITextDetector class
# Remove Flask imports
from queue import Queue
from threading import Lock
from PyQt5.QtWidgets import QMessageBox

try:
    from infer_model import AIPlagiarismDetector as AdvancedAIPlagiarismDetector
    from infer_model2 import AIPlagiarismDetector as BasicAIPlagiarismDetector
except ImportError as e:
    print(f"Import error: {e}")
    # Ensure at least the basic detector is available
    from infer_model2 import AIPlagiarismDetector as BasicAIPlagiarismDetector

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

        # Initially hide batch operation buttons and batch size labels
        self.ui.btnBatchTake.hide()
        self.ui.btnBatchNext.hide()
        self.ui.btnBatchStop.hide()
        self.ui.lblBatchSize.hide()
        self.ui.lblBatchSizeVal.hide()
        
        # Initialize the advanced AI detector
        try:
            print("Attempting to initialize advanced detector...")
            self.detector = AdvancedAIPlagiarismDetector(
                'model/trained_model3.pkl',
                'model/trained_model3_xgb.pkl'  # Second model is optional
            )
            print("Advanced detector initialized successfully")
        except Exception as e:
            print(f"Error initializing advanced detector: {e}")
            print("Falling back to basic detector...")
            # Fallback to the simpler detector if the advanced one fails
            try:
                self.detector = BasicAIPlagiarismDetector('model/trained_model2.pkl')
                print("Basic detector initialized successfully")
            except Exception as e2:
                print(f"Error initializing basic detector: {e2}")
                # Last resort fallback
                from infer_model import AIPlagiarismDetector
                self.detector = AIPlagiarismDetector('model/trained_model1.pkl')
                print("Fallback detector initialized")

    def show_message_dialog(self, title, message, icon=QMessageBox.Information):
        """Display a message dialog with the given title and message"""
        msg_dialog = QMessageBox(self)
        msg_dialog.setIcon(icon)
        msg_dialog.setWindowTitle(title)
        msg_dialog.setText(message)
        msg_dialog.setStandardButtons(QMessageBox.Ok)
        msg_dialog.exec_()

    def show_progress_dialog(self, title, message):
        """Display a non-blocking progress message"""
        # Close any existing progress dialog first
        self.close_progress_dialog()
        
        # Create and show new progress dialog
        self.progress_dialog = QMessageBox(self)
        self.progress_dialog.setIcon(QMessageBox.Information)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setText(message)
        self.progress_dialog.setStandardButtons(QMessageBox.NoButton)
        
        # Make dialog closable by clicking the X button
        self.progress_dialog.setWindowFlags(
            self.progress_dialog.windowFlags() | 
            QtCore.Qt.WindowCloseButtonHint
        )
        
        # Connect close event to our close_progress_dialog method
        self.progress_dialog.closeEvent = lambda event: self.close_progress_dialog()
        
        # Show the dialog without blocking
        self.progress_dialog.show()
        
        # Process events to ensure the dialog is displayed immediately
        QtWidgets.QApplication.processEvents()
    
    def close_progress_dialog(self):
        """Close the progress dialog if it exists"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            try:
                self.progress_dialog.close()
            except Exception as e:
                print(f"Error closing progress dialog: {e}")
            finally:
                self.progress_dialog = None
        
        # Process events to ensure UI updates
        QtWidgets.QApplication.processEvents()

    def toggle_batch_mode(self):
        """Toggle between single capture and batch capture modes"""
        self.batch_mode = True
        
        # Show batch operation buttons, hide single capture button
        self.ui.btnSingleCap.hide()
        self.ui.btnBatchCap.hide()
        self.ui.btnBatchTake.show()
        self.ui.btnBatchNext.show()
        self.ui.btnBatchStop.show()
        
        # Show batch size labels
        self.ui.lblBatchSize.show()
        self.ui.lblBatchSizeVal.show()
        self.ui.lblBatchSizeVal.setText("0")  # Reset batch size counter
        
        # Clear any existing queue
        with self.queue_lock:
            while not self.image_queue.empty():
                self.image_queue.get()
        
        print("Batch mode activated. Capture images and then process them.")
        
        # Show a message dialog instead of using resultTextEdit
        self.show_message_dialog(
            "Batch Mode", 
            "Batch mode activated.\n\n"
            "1. Click 'Capture' to add images to the batch.\n"
            "2. When all pages are captured, click 'Process' to analyze the entire document.\n"
            "3. Click 'Stop' to exit batch mode without processing."
        )
    
    def process_batch(self):
        """Process all images in the queue as a single document"""
        
        with self.queue_lock:
            queue_size = self.image_queue.qsize()
            
            if queue_size == 0:
                print("No images in batch to process")
                self.show_error_dialog("Batch Processing Error", "No images in batch to process")
                return
            
            # Show progress dialog
            self.show_progress_dialog("Processing Batch", 
                                     f"Processing {queue_size} images as a single document...\n"
                                     f"AI plagiarism detection in progress. Please wait.")
            
            # Force UI update to ensure dialog appears immediately
            QtWidgets.QApplication.processEvents()
            
            print(f"Processing {queue_size} images as a single document")
            
            # Extract text from all images and concatenate
            all_extracted_text = ""
            image_count = 0
            error_occurred = False
            error_message = ""
            
            # First pass: extract text from all images
            temp_queue = Queue()
            
            while not self.image_queue.empty():
                try:
                    image_array = self.image_queue.get()
                    temp_queue.put(image_array)  # Save image for potential debugging
                    
                    # Update progress dialog with current image being processed
                    if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                        self.progress_dialog.setText(f"Processing image {image_count+1} of {queue_size}...\n"
                                                   f"Extracting text. Please wait.")
                        QtWidgets.QApplication.processEvents()
                    
                    # Extract text from the image
                    extractor = ImageTextExtractor(image_array)
                    extracted_text = extractor.extract_text()
                    
                    # Skip if no text was extracted
                    if not extracted_text or len(extracted_text.strip()) < 5:
                        print(f"No meaningful text extracted from image {image_count+1}, skipping")
                        continue
                    
                    # Add page number and append to the combined text
                    all_extracted_text += f"\n\n--- Page {image_count+1} ---\n\n"
                    all_extracted_text += extracted_text
                    image_count += 1
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Error extracting text from image: {e}")
                    error_occurred = True
                    continue
            
            # Check if we have any text to process
            if not all_extracted_text or len(all_extracted_text.strip()) < 10:
                self.close_progress_dialog()
                error_msg = "No meaningful text extracted from any images in the batch"
                if error_occurred:
                    error_msg += f"\nError: {error_message}"
                
                print(error_msg)
                self.show_error_dialog("Batch Processing Failed", error_msg)
                self.stop_batch_mode()
                return
            
            # Process the combined text with the AI detector
            try:
                # Update progress dialog for AI detection
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    self.progress_dialog.setText(f"Analyzing combined text from {image_count} images...\n"
                                               f"AI plagiarism detection in progress. Please wait.")
                    QtWidgets.QApplication.processEvents()
                
                print(f"Analyzing combined text from {image_count} images")
                result = self.detector.detect_ai_text(all_extracted_text)
                
                # Close progress dialog
                self.close_progress_dialog()
                
                # Display results in a message box (without text preview)
                result_text = f"Document Analysis Complete\n\n"
                result_text += f"Images processed: {image_count}\n"
                result_text += f"Result: {result[0]}\n"
                result_text += f"Perplexity: {result[1]:.2f}\n"
                result_text += f"Burstiness: {result[2]:.2f}\n\n"
                result_text += f"Interpretation: {result[3] if len(result) > 3 else 'No interpretation available'}"
                
                self.show_message_dialog("Batch Analysis Results", result_text)
                print(f"Batch processing complete. Result: {result[0]}")
                
            except Exception as e:
                self.close_progress_dialog()
                error_msg = f"Error analyzing combined text: {str(e)}"
                print(error_msg)
                self.show_error_dialog("Analysis Error", error_msg)
        
        # Return to normal mode after processing
        self.stop_batch_mode()
    
    def stop_batch_mode(self):
        """Exit batch mode and return to normal operation"""
        self.batch_mode = False
        
        # Show normal buttons, hide batch operation buttons
        self.ui.btnSingleCap.show()
        self.ui.btnBatchCap.show()
        self.ui.btnBatchTake.hide()
        self.ui.btnBatchNext.hide()
        self.ui.btnBatchStop.hide()
        
        # Hide batch size labels
        self.ui.lblBatchSize.hide()
        self.ui.lblBatchSizeVal.hide()
        
        print("Batch mode deactivated.")

    def load_file(self):
        # Get the file path from the lineEdit widget
        file_path = self.ui.lineEdit.text()

        # Show progress dialog immediately
        self.show_progress_dialog("Processing File", 
                                 "Loading file...\nPlease wait.")
        QtWidgets.QApplication.processEvents()

        try:
            # Load the image using OpenCV
            image_array = cv2.imread(file_path)

            if image_array is None:
                self.close_progress_dialog()
                raise FileNotFoundError("Could not load the image from the provided path.")

            # Update progress dialog
            if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                self.progress_dialog.setText("Extracting text from file...\nPlease wait.")
                QtWidgets.QApplication.processEvents()

            # Process the loaded image
            extractor = ImageTextExtractor(image_array)
            extracted_text = extractor.extract_text()
            print("Extracted Text from File:")
            print(extracted_text)
            
            # Update progress dialog
            if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                self.progress_dialog.setText("AI plagiarism detection in progress...\nPlease wait.")
                QtWidgets.QApplication.processEvents()
            
            # Detect AI-generated text
            result = self.detector.detect_ai_text(extracted_text)
            print(result)

            # Close progress dialog
            self.close_progress_dialog()

            # Display results in a message box (without text preview)
            result_text = f"File: {file_path}\n\n"
            result_text += f"Result: {result[0]}\n"
            result_text += f"Perplexity: {result[1]:.2f}\n"
            result_text += f"Burstiness: {result[2]:.2f}\n\n"
            result_text += f"Interpretation: {result[3]}"
            
            self.show_message_dialog("File Analysis Results", result_text)
            
        except Exception as e:
            self.close_progress_dialog()
            error_msg = f"Error loading file: {e}"
            print(error_msg)
            self.show_error_dialog("File Loading Error", error_msg)

    def capture_image(self):
        # If not in batch mode, show progress dialog immediately
        if not self.batch_mode:
            self.show_progress_dialog("Processing", "Capturing image...\nPlease wait.")
        
        # Create and start the capture thread
        self.capture_thread = CaptureThread(self.picam2)
        self.capture_thread.capture_done.connect(self.on_capture_done)
        self.capture_thread.start()

    def on_capture_done(self, image_array):
        """Handle captured image based on current mode"""
        
        if self.batch_mode:
            # In batch mode, add to queue instead of processing immediately
            with self.queue_lock:
                self.image_queue.put(image_array)
                queue_size = self.image_queue.qsize()
            
            # Update batch size label
            self.ui.lblBatchSizeVal.setText(str(queue_size))
        
            print(f"Added image to batch. Queue size: {queue_size}")
            # Remove the message dialog - the label is sufficient feedback
        else:
            # In single capture mode, process immediately as before
            try:
                # Update progress dialog message
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    self.progress_dialog.setText("Extracting text from image...\nPlease wait.")
                    QtWidgets.QApplication.processEvents()
                else:
                    # If dialog was closed, create a new one
                    self.show_progress_dialog("Processing", "Extracting text from image...\nPlease wait.")
                
                extractor = ImageTextExtractor(image_array)
                extracted_text = extractor.extract_text()
                
                if not extracted_text or len(extracted_text.strip()) < 5:
                    self.close_progress_dialog()  # Ensure dialog is closed
                    error_msg = "No meaningful text extracted from image"
                    print(error_msg)
                    self.show_error_dialog("Text Extraction Error", error_msg)
                    return
                
                # Update progress dialog for AI detection step
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    self.progress_dialog.setText("AI plagiarism detection in progress...\nPlease wait.")
                    QtWidgets.QApplication.processEvents()
                    
                print("Extracted Text:")
                print(extracted_text)
                
                # Use AITextDetector to check if the extracted text is AI-generated
                result = self.detector.detect_ai_text(extracted_text)
                print(result)

                # Close progress dialog before showing results
                self.close_progress_dialog()

                # Display results in a message box (without text preview)
                result_text = f"Result: {result[0]}\n"
                result_text += f"Perplexity: {result[1]:.2f}\n"
                result_text += f"Burstiness: {result[2]:.2f}\n\n"
                result_text += f"Interpretation: {result[3] if len(result) > 3 else 'No interpretation available'}"
                
                self.show_message_dialog("Analysis Results", result_text)
                
            except Exception as e:
                # Ensure dialog is closed even on exception
                self.close_progress_dialog()
                error_msg = f"Error processing image: {str(e)}"
                print(error_msg)
                self.show_error_dialog("Processing Error", error_msg)

# Remove run_flask function

if __name__ == "__main__":
    # Remove Flask thread
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
