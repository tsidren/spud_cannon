import cv2
import numpy as np

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

# Example usage
# Load the image from the camera
image = cv2.imread('abc.jpg')

# Detect green objects and draw bounding boxes
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
        image_width = image.shape[1]
        angle = (center_x - image_width / 2) * 45 / (image_width / 2)

        # Display the selected box's information in the same window
        output_image = annotated_image.copy()
        cv2.putText(output_image, "Center Pixel: ({}, {})".format(center_x, center_y), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(output_image, "Angle: {:.2f} degrees".format(angle), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Annotated Image", output_image)

cv2.destroyAllWindows()