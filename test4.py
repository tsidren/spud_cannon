import socket
import struct
import numpy as np
import cv2

# ESP32-CAM server parameters
server_ip = '162.16.1.159'  # Replace with the ESP32-CAM server IP address
server_port = 80


# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 30])
#
# lower_blue = np.array([90, 50, 50])
# upper_blue = np.array([130, 255, 255])

def detect_green_objects(image):
    # Convert image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a binary mask for the green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours based on their area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a list to store the bounding box coordinates
    bounding_boxes = []

    for contour in contours[:6]:
        # Calculate the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        # Draw a bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(len(bounding_boxes)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return bounding_boxes, image


def receive_image_and_send_coordinates():
    # Connect to the ESP32-CAM server
    print("here 0")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    print("here 1")

    # Receive the image size
    image_size_bytes = client_socket.recv(4)
    image_size = struct.unpack('i', image_size_bytes)[0]
    print("Received image size:", image_size)

    # Receive the image data
    image_data = b''
    while len(image_data) < image_size:
        chunk = client_socket.recv(image_size - len(image_data))
        if not chunk:
            break
        image_data += chunk
    print("here 3")

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Reshape the image array based on the desired resolution
    image = image_array.reshape((480, 640, 3))

    # Process the image using OpenCV (replace with your target detection logic)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
    bounding_boxes, annotated_image = detect_green_objects(image.copy())

    # Create a separate window for displaying the annotated image
    cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Annotated Image", annotated_image)

    # Wait for the user to provide input and display the output in the same window
    while True:
        key = cv2.waitKey(1)

        if key == 27:  # Check if the "Esc" key is pressed
            break

        # Ask for the input of the box number
        box_number = int(cv2.waitKey(0)) - ord('0')

        if box_number >= 1 and box_number <= len(bounding_boxes):
            # Get the selected box's coordinates
            x, y, w, h = bounding_boxes[box_number - 1]

            # Calculate the center of the box
            center_x = x + (w // 2)
            center_y = y + (h // 2)

            # Calculate the direction in degrees
            # image_width = image.shape[1]
            # angle = (center_x - image_width / 2) * 45 / (image_width / 2)

    cv2.destroyAllWindows()
    # //////////////////////////////////////////////////////////////////////////////////////////////////////
    # Perform additional processing on the image as needed

    # Send the coordinates to the ESP32-CAM server
    x_bytes = struct.pack('i', center_x)
    client_socket.sendall(x_bytes)
    y_bytes = struct.pack('i', center_y)
    client_socket.sendall(y_bytes)

    # Close the connection
    client_socket.close()

# Call the function to initiate the communication
receive_image_and_send_coordinates()
