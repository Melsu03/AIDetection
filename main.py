import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal

from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2

from QtMainWindow import QtMainWindow

class CaptureThread(QThread):
    capture_done = pyqtSignal(str)

    def __init__(self, picam2, image_path):
        super().__init__()
        self.picam2 = picam2
        self.image_path = image_path

    def run(self):
        self.picam2.switch_mode_and_capture_file(self.picam2.create_still_configuration(), self.image_path)
        self.capture_done.emit(self.image_path)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.ui = QtMainWindow()
        self.ui.setupUi(self)

        # Initialize Picamera2
        self.picam2 = Picamera2()

        # Set up camera preview configuration
        preview_width = 640
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

        # Connect capture button to the capture function
        self.ui.btnCapture.clicked.connect(self.capture_image)

    def capture_image(self):
        # Capture an image and save it
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd-HHmmss")
        image_path = f"/home/rpiuser/source/AIDetection/cam/img_{timestamp}.jpg"
        print(f"Capturing image to {image_path}")

        # Create and start the capture thread
        self.capture_thread = CaptureThread(self.picam2, image_path)
        self.capture_thread.capture_done.connect(self.on_capture_done)
        self.capture_thread.start()

    def on_capture_done(self, image_path):
        print(f"Image captured to {image_path}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())