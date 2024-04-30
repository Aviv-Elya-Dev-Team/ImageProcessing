import cv2
import numpy as np, numpy


base_images = {
    2: cv2.imread("imgs/2.jpg")[1800:3400, 900:2500],
    5: cv2.imread("imgs/5.jpg")[2025:2925, 1175:2075],
    10: cv2.imread("imgs/10.jpg")[2160:3060, 1310:2210],
    50: cv2.imread("imgs/50.jpg")[2235:3320, 865:2000],
}


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
    return radius / (50 + distance(center_img, (circle_x, circle_y)))


def distance(point1, point2):
    # Euclidean distance
    x1, y1 = point1
    x2, y2 = point2
    distance = numpy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance


def mask_circular_objects(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=200,
        maxRadius=400,
    )

    # Initialize a mask with zeros
    mask = np.zeros(gray.shape, dtype=np.uint8)

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.round(circles[0, :]).astype("int")

        # Draw circles on the mask
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, (255), -1)

    return mask


def main():
    mask_coin(cv2.imread("imgs/62.jpg"))


if __name__ == "__main__":
    main()
