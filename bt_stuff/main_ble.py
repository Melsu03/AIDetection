import sys
import subprocess
import time
import json
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from picamera2 import Picamera2, controls
from picamera2.previews.qt import QGlPicamera2
import numpy as np
from ui_main_window import QtMainWindow
from ocr import ImageTextExtractor  # Import the ImageTextExtractor class
from infer_model import AIPlagiarismDetector  # Import the AITextDetector class

class CaptureThread(QThread):
    capture_done = pyqtSignal(np.ndarray)

    def __init__(self, picam2):
        super().__init__()
        self.picam2 = picam2

    def run(self):
        # Capture the image directly into a NumPy array
        image_array = self.picam2.capture_array()
        self.capture_done.emit(image_array)
        
class FileWatcherThread(QThread):
    connection_status = pyqtSignal(str)  # Signal to notify about connection status
    ble_data = pyqtSignal(str)  # Signal to notify about BLE data

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """Watch the file for changes and emit status updates."""
        last_position = 0
        while True:
            try:
                with open(self.file_path, "r") as file:
                    file.seek(last_position)
                    new_content = file.read()
                    if new_content:
                        last_position = file.tell()
                        status = json.loads(new_content)  # Parse the JSON content
                        self.connection_status.emit(status["status"])  # Emit the current connection status
                        print(status["status"])
                        self.ble_data.emit(status["bledata"])  # Emit the latest BLE data
                        print(status["bledata"])
            except (FileNotFoundError, json.JSONDecodeError):
                # Handle the case where the file doesn't exist or is invalid
                pass
            time.sleep(1)  # Check for new content every second

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.ui = QtMainWindow()
        self.ui.setupUi(self)
        
        # Start file watcher thread
        self.file_watcher = FileWatcherThread("bluetooth_data.json")
        self.file_watcher.connection_status.connect(self.on_bluetooth_status)
        #self.file_watcher.ble_data.connect(self.on_ble_data)
        self.file_watcher.start()
        
        # Connect to the BLE server's stdout (for receiving status)
        self.ui.btnCapture.setEnabled(False)  # Disable until Bluetooth is connected

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
        self.ui.btnCapture.clicked.connect(self.capture_image)
        
        # Initialize the AITextDetector
        self.detector = AIPlagiarismDetector('model/trained_model1.pkl')

    def capture_image(self):
        # Capture an image and save it
        # timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd-HHmmss")
        # image_path = f"/home/rpiuser/source/AIDetection/cam/img_{timestamp}.jpg"
        # print(f"Capturing image to {image_path}")

        # Create and start the capture thread
        self.capture_thread = CaptureThread(self.picam2)
        self.capture_thread.capture_done.connect(self.on_capture_done)
        self.capture_thread.start()

    def on_capture_done(self, image_array):
        # print(f"Image captured to {image_path}")
        
        # Use ImageTextExtractor to extract text from the image array
        extractor = ImageTextExtractor(image_array)
        extracted_text = extractor.extract_text()
        print("Extracted Text:")
        print(extracted_text)
        
        # Use AITextDetector to check if the extracted text is AI-generated
        result = self.detector.detect_ai_text(extracted_text)
        print(result)

    def on_bluetooth_status(self, status):
        """Update the Bluetooth connection status in the UI."""
        self.ui.lblBLEWait.setText(f"Bluetooth Status: {status}")
        if status == "Connected":
            self.ui.btnCapture.setEnabled(True)  # Enable capture button when connected
        elif status == "Disconnected":
            self.ui.btnCapture.setEnabled(False)  # Disable capture button when disconnected

    def closeEvent(self, event):
        """Ensure to terminate the subprocess when the app closes."""
        self.file_watcher.terminate()  # Stop file watcher thread
        self.file_watcher.wait()  # Wait for it to shut down properly
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
