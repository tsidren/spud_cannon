import requests

# ESP32-CAM module IP address and port
module_ip = "162.16.1.159"  # Replace with the actual IP address of your module
module_port = 8080  # Replace with the port number used in the ESP32-CAM program

# Message to send
message = "Hello, ESP32-CAM!"

# URL for the message endpoint
url = f"http://{module_ip}:{module_port}/message"

# Send the message to the ESP32-CAM module
try:
    response = requests.post(url, data=message)
    response.raise_for_status()
    print("Message sent successfully")
except requests.exceptions.RequestException as e:
    print(f"Error sending message: {str(e)}")
