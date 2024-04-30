import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image_plt(image):
    plt.imshow(image, cmap="gray")
    plt.show()

# Read the image
image = cv2.imread('10.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to obtain a binary image
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

# Apply morphological dilation to fill the circles
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)

# Invert the binary image
filled_circles = cv2.bitwise_not(dilated)

# Display the result
show_image_plt(filled_circles)
