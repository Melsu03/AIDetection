import subprocess
from PyQt5 import QtWidgets, QtCore
import sys

class BluetoothServer(QtCore.QThread):
    device_found = QtCore.pyqtSignal(str)  # Signal to notify when a device is found
    status_updated = QtCore.pyqtSignal(str)  # Signal to update Bluetooth status

    def __init__(self):
        super().__init__()

    def run(self):
        # Start bluetoothctl as a subprocess and interact with it
        process = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Make the Raspberry Pi discoverable (optional)
        process.stdin.write("power on\n")
        process.stdin.write("agent on\n")
        process.stdin.write("default-agent\n")
        process.stdin.write("discoverable on\n")
        process.stdin.write("pairable on\n")
        process.stdin.flush()

        self.status_updated.emit("Bluetooth is active. Waiting for devices...")

        # Continuously check for any Bluetooth activity
        while True:
            output = process.stdout.readline()
            if output == '':
                break
            print(f"Bluetooth output: {output.strip()}")
            if "Device found" in output:
                self.device_found.emit(output.strip())  # Emit signal when device is found

        process.stdin.close()
        process.stdout.close()
        process.stderr.close()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bluetooth Server UI")
        self.setGeometry(100, 100, 300, 200)

        # Create a label to show the Bluetooth connection status
        self.status_label = QtWidgets.QLabel("Bluetooth Status: Waiting...", self)
        self.status_label.setGeometry(50, 50, 200, 30)

        # Create a label to show the found device
        self.device_label = QtWidgets.QLabel("Device: None", self)
        self.device_label.setGeometry(50, 100, 200, 30)

        # Initialize the Bluetooth server in a separate thread
        self.bluetooth_server = BluetoothServer()
        self.bluetooth_server.device_found.connect(self.on_device_found)
        self.bluetooth_server.status_updated.connect(self.on_status_updated)
        self.bluetooth_server.start()

    def on_device_found(self, device_info):
        self.device_label.setText(f"Device: {device_info}")
        print(f"Device found: {device_info}")

    def on_status_updated(self, status):
        self.status_label.setText(f"Bluetooth Status: {status}")
        print(f"Status updated: {status}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
