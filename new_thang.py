import requests

# ESP32-CAM server IP address
server_ip = '162.16.1.159'  # Replace with the actual IP address of your ESP32-CAM module

# Text data to send
message = 'Hello, ESP32-CAM!'

# Send the message to the ESP32-CAM server
response = requests.post(f'http://{server_ip}/message', data=message)

# Check the response
if response.status_code == 200:
    print('Message sent successfully')
else:
    print('Failed to send message')
