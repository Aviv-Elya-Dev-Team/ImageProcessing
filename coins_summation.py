import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from coin_radius_finder import mask_coin, base_images, mask_circular_objects

# typing
import numpy.typing

# from cv2.typing import Point # TODO: remove if unused

MatLike = numpy.typing.NDArray[numpy.uint8]


def show_image_cv(img, title="title"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, len(img[0]), len(img))
    cv2.imshow(title, img)
    cv2.waitKey(0)


def show_image_plt(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def clean_image(image, size=(11, 11), sigmaX=5):
    return cv2.GaussianBlur(image, ksize=size, sigmaX=sigmaX)


def calculate_masked_images():
    masked_images = {}
    load = False
    for key in [2, 5, 10, 50]:
        if os.path.exists(f"templates/{str(key)}.png"):
            masked_images[key] = cv2.imread(f"templates/{str(key)}.png")
            load = True

    if load:
        return masked_images
    else:
        print("calculating base images masks, please wait...")

    masked_images = {
        2: mask_coin(base_images[2]),
        5: mask_coin(base_images[5]),
        10: mask_coin(base_images[10]),
        50: mask_coin(base_images[50]),
    }
    for key in masked_images.keys():
        cv2.imwrite(f"templates/{str(key)}.png", masked_images[key])
    print("done calculating masks")
    return masked_images


def count_matches(template_image: MatLike, target_image: MatLike, threshold=0.8):
    """
    counts how many good matches are in the target_image from the template_image

    Args:
        template_image (MatLike): input image to extract matches from and search in target_image
        target_image (MatLike): image to find matches in
        threshold (float, optional): filters out matches too close to each other. Defaults to 0.8.

    Returns:
        int: number of good matches (within the threshold) found
    """

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors for both images
    keypoints_template, descriptors_template = sift.detectAndCompute(
        template_image, None
    )
    keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)

    # Initialize a FLANN matcher
    flann = cv2.FlannBasedMatcher()

    # Perform KNN matching to find the best matches
    matches = flann.knnMatch(descriptors_template, descriptors_target, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    return len(good_matches)

    # get the location of keypoints in the target image
    target_keypoint_locations = [
        keypoints_target[match.trainIdx].pt for match in good_matches
    ]
    # count the number of unique occurrences
    unique_locations = set(target_keypoint_locations)
    return len(unique_locations)


def sum_coins(masked_images, target_image):
    result = 0

    # extract just the coins from the target image
    target_coins_circles = find_coins(target_image)
    target_coins_images = [
        extract_coin_from_image(target_image, (coin[0], coin[1]), coin[2])
        for coin in target_coins_circles
    ]

    # classify each coin from the target image and add to the sum
    for target_coin_image in target_coins_images:
        target_coin = classify_from_image(masked_images, target_coin_image)
        result += target_coin

    return result


def classify_from_image(masked_images: dict, coin_image: MatLike) -> int:
    """
    checks which coin the given coin image most closely resembles
    from the base and masked coin images (2, 5, 10, 50)

    Args:
        coin_image (MatLike): the input coin image

    Returns:
        int: the coin that the given image represents the most
    """
    result = -1
    max_matches = 0
    for base_coin in masked_images.keys():
        num_matches = count_matches(masked_images[base_coin], coin_image)
        if num_matches > max_matches:
            result = base_coin
            max_matches = num_matches

    return result


def circle_edges(image):
    # Read the image
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (255, 255), 0)
    kernel = np.ones((11, 11), dtype=np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)
    edges = dilated - blurred
    _, edges = cv2.threshold(edges, 3, 255, cv2.THRESH_BINARY)
    edges = cv2.erode(edges, kernel, iterations=3)
    return edges


def find_coins(image, threshold=200, draw_circles_on_image=False):
    """
    uses hough transform with circles to detect coins inside the image.
    then, it extracts their center and radius and returns a list of all of em
    (not perfect but it works pretty good)

    Args:
        image: _description_
        threshold (int, optional): _description_. Defaults to 200.

    Returns:
        list: a list that contains the coordiantes of the center of the coin and its radius
    """

    # make the image gray if its not already
    gray = image
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = circle_edges(gray)
    circles = circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0
    )
    circle_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        grouped_circles = {}
        for x, y, r in circles:
            added_to_group = False
            for center_point, circle_group in grouped_circles.items():
                if (
                    np.linalg.norm(np.array([x, y]) - np.array(center_point))
                    < threshold
                ):
                    circle_group.append((x, y, r))
                    added_to_group = True
                    break
            if not added_to_group:
                grouped_circles[(x, y)] = [(x, y, r)]

        # calculate the average center point and radius for each group
        # and append to circle list
        for center_point, circle_group in grouped_circles.items():
            group_x, group_y, group_r = zip(*circle_group)
            avg_x = int(np.mean(group_x))
            avg_y = int(np.mean(group_y))
            avg_r = int(np.mean(group_r))
            circle_list.append([avg_x, avg_y, avg_r])

        # draw the average circles on the original image if the user wants
        if draw_circles_on_image:
            for avg_x, avg_y, avg_r in circle_list:
                cv2.circle(image, (avg_x, avg_y), avg_r, (0, 0, 255), 4)

    return circle_list


def extract_coin_from_image(image, coin_center, coin_radius):
    center_x, center_y = coin_center

    # calculate box boundaries
    top_left_x = int(center_x - coin_radius)
    top_left_y = int(center_y - coin_radius)
    bottom_right_x = int(center_x + coin_radius)
    bottom_right_y = int(center_y + coin_radius)

    # take care of edges of the image
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(image.shape[1], bottom_right_x)
    bottom_right_y = min(image.shape[0], bottom_right_y)

    # extract the circle from the image and
    # create a new image with just the coin
    circle_region = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    mask = np.zeros_like(circle_region)
    cv2.circle(mask, (coin_radius, coin_radius), coin_radius, (255, 255, 255), -1, 8, 0)

    masked_circle = cv2.bitwise_and(circle_region, mask)
    result = np.zeros_like(masked_circle)
    result[mask == 255] = masked_circle[mask == 255]

    return result


def main():
    masked_images = calculate_masked_images()

    target_image = cv2.imread("imgs/144_2.jpg")

    print(sum_coins(masked_images, target_image))
    # print(count_object_appearances(template_image, target_image))


if __name__ == "__main__":
    main()
