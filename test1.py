import cv2
import numpy as np

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

    # Create a list to store the bounding box coordinates
    bounding_boxes = []

    for contour in contours:
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
bounding_boxes, annotated_image = detect_green_objects(image)

# Display the annotated image
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)

# Ask for the input of the box number
box_number = int(input("Enter the box number: "))

if box_number >= 1 and box_number <= len(bounding_boxes):
    # Get the selected box's coordinates
    x, y, w, h = bounding_boxes[box_number - 1]

    # Calculate the center of the box
    center_x = x + (w // 2)
    center_y = y + (h // 2)

    # Calculate the direction in degrees
    image_width = image.shape[1]
    angle = (center_x - image_width / 2) * 45 / (image_width / 2)

    print("Selected Box Center Pixel:", (center_x, center_y))
    print("Direction (Angle):", angle, "degrees")
else:
    print("Invalid box number.")
