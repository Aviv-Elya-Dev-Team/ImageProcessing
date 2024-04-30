import cv2
import numpy as np, numpy
from coins_summation import show_image_plt
from coins_summation import base_images

def mask_coin(image):
    height, width, _ = image.shape

    # work with grayscale to reduce calculations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # use hough transform on circles to find the coin outer circle
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=150,
        param1=50,
        param2=150,
        minRadius=300,
        maxRadius=1000,
    )

    if circles is not None:
        image_center = (width // 2, height // 2)
        best_score = 0
        best_circle = None
        # iterate over all the detected circles and find best ones using circle_score
        for circle in circles[0, :]:
            # choose best circle by its score
            score = circle_score(circle, image_center)
        
            if score > best_score:
                best_circle = circle
                best_score = score

        # create a mask to keep only the area inside the closest circle (zero out everything else)
        mask = np.zeros_like(gray)
        cv2.circle(
            mask,
            (int(best_circle[0]), int(best_circle[1])),
            int(best_circle[2]),
            (255, 255, 255),
            -1,
        )

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

    return result

# get score of the circle, we want bigger
# radius and small distance from the center
def circle_score(circle, center_img):
    circle_x, circle_y, radius = circle
    return radius/(50 + distance(center_img, (circle_x, circle_y)))

def distance(point1, point2):
    # Euclidean distance
    x1, y1 = point1
    x2, y2 = point2
    distance = numpy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance  

def main():
    show_image_plt(mask_coin(base_images[2]))
    show_image_plt(mask_coin(base_images[5]))
    show_image_plt(mask_coin(base_images[10]))
    show_image_plt(mask_coin(base_images[50]))

if __name__ == "__main__":
    main()
