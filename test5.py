import requests
import cv2
import numpy as np

# ESP32-CAM server IP address
server_ip = 'xxx.xxx.xxx.xxx'  # Replace with the actual IP address of your ESP32-CAM module

# Send a request to the ESP32-CAM server to capture an image
response = requests.get(f'http://{server_ip}/capture')

# Read the response content as bytes
image_data = response.content

# Convert the image data to a NumPy array
image_array = np.frombuffer(image_data, dtype=np.uint8)

# Decode the image array into OpenCV format
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Display the image
cv2.imshow('Captured Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
