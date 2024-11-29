import btfpy
import time
import json

# Path to the JSON status file
status_file_path = "bluetooth_data.json"

# Function to read and write to the JSON status file
def read_status_file():
    try:
        with open(status_file_path, "r") as file:
            status = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"status": "Disconnected", "bledata": "", "input": ""}  # Default values
    return status

def write_status_file(status_data):
    with open(status_file_path, "w") as file:
        json.dump(status_data, file, indent=4)

def callback(clientnode, operation, cticn):
    # Read current status from the file
    status = read_status_file()

    if operation == btfpy.LE_CONNECT:
        status["status"] = "Connected"
        print("Connected")
    elif operation == btfpy.LE_DISCONNECT:
        status["status"] = "Disconnected"
        print("Disconnected")
        return btfpy.SERVER_EXIT
    elif operation == btfpy.LE_TIMER:
        print("BLE server is running.")
        print(btfpy.Device_connected(btfpy.Localnode()))
        
        # Periodic action: Write data to the characteristic and read it back
        btfpy.Write_ctic(btfpy.Localnode(), 1, status["input"], 0)
        bledata = btfpy.Read_ctic(btfpy.Localnode(), 2)
        
        # Decode bytes to a string for JSON compatibility
        bledata_str = bledata.decode('utf-8') if isinstance(bledata, bytes) else str(bledata)
        status["bledata"] = bledata_str  # Save the latest data as a string in the status file

        # Write updated status back to the file
        write_status_file(status)
        
    return btfpy.SERVER_CONTINUE

def start_ble_server():
    # Initialize Bluetooth
    if btfpy.Init_blue("devices.txt") == 0:
        print("Error initializing Bluetooth")
        return

    btfpy.Le_server(callback, 10)  # 50ms for timer checks, adjust as necessary
    btfpy.Close_all()
    # Run the BLE server
    #try:
    #    while True:
    #        btfpy.Le_server(callback, 50)  # 50ms for timer checks, adjust as necessary
    #except Exception as e:
    #    print(f"Error with server: {e}")
    #finally:
    #    btfpy.Close_all()

if __name__ == "__main__":
    start_ble_server()
