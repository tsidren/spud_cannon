import cv2
import numpy as np

def detect_target(image):
    # Convert image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a binary mask for the light green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the contour
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            # Calculate the centroid of the contour
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

            # Calculate the direction in degrees
            image_width = image.shape[1]
            angle = (cX - image_width / 2) * 45 / (image_width / 2)

            return (cX, cY), angle

    return None, None

# Example usage
# Load the image from the camera
image = cv2.imread('abc.jpg')

# Detect the target
target_center, target_angle = detect_target(image)

if target_center is not None:
    print("Target Center Pixel:", target_center)
    print("Direction (Angle):", target_angle, "degrees")
else:
    print("Target not found.")
