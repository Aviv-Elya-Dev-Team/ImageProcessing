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


def show_image_plt(image, title="Figure"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
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
            2: base_images[2],
            5: base_images[5],
            10: base_images[10],
            50: base_images[50]
        }
        # blur the 50,5 coin because its too high quality and it fucks with
        # finding keypoint descriptors later
        # masked_images[50] = cv2.GaussianBlur(masked_images[50], (5,5), 2)
        # masked_images[10] = cv2.GaussianBlur(masked_images[10], (3,3), 0.67)
        # masked_images[5] = cv2.GaussianBlur(masked_images[5], (5,5), 1.9)
        # masked_images[2] = cv2.GaussianBlur(masked_images[2], (3,3), 0.37)

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
    """
    matched_img = cv2.drawMatches(
        template_image,
        keypoints_template,
        target_image,
        keypoints_target,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    show_image_plt(matched_img)
    return len(good_matches)
    """
    # get the location of keypoints in the target image
    target_keypoint_locations = [
        keypoints_target[match.trainIdx].pt for match in good_matches
    ]
    # count the number of unique occurrences
    unique_locations = set(target_keypoint_locations)
    # cv2.drawKeypoints(image, target_keypoint_locations)
    return len(unique_locations)


def sum_coins(masked_images, target_image):
    base_coin_matches = {
        2: count_matches(masked_images[2], masked_images[2]),
        5: count_matches(masked_images[5], masked_images[5]),
        10: count_matches(masked_images[10], masked_images[10]),
        50: count_matches(masked_images[50], masked_images[50]),
    }
    print(base_coin_matches)
    result = 0

    # extract just the coins from the target image
    target_coins_circles = find_coins(target_image)
    target_coins_images = [
        extract_coin_from_image(target_image, (coin[0], coin[1]), coin[2])
        for coin in target_coins_circles
    ]

    # classify each coin from the target image and add to the sum
    for target_coin_image in target_coins_images:
        target_coin = find_best_matching_template(masked_images, target_coin_image)
        """"
        target_coin = classify_from_image(
            masked_images, base_coin_matches, target_coin_image
        )
        """
        result += target_coin
        show_image_plt(
            target_coin_image, title=f"Prediction: {target_coin} Sum: {result}"
        )
    return result


def classify_from_image(
    masked_images: dict, base_coin_matches: dict, coin_image: MatLike, threshold=0
) -> int:
    """
    checks which coin the given coin image most closely resembles
    from the base and masked coin images (2, 5, 10, 50)

    Args:
        masked_images (dict): the base coin images
        base_coin_matches (dict): the values of matches for each base coin with itself
        coin_image (MatLike): the input coin image
        threshold (int): will ignore coins that don't get
                         any matches above this threshold with all the base coins

    Returns:
        int: the coin that the given image represents the most
    """

    def ratio(num_matches):
        return num_matches  # 9 ** (numpy.log10(num_matches))

    result = 0
    max_matches = 0
    for base_coin in masked_images.keys():
        num_matches = count_matches(masked_images[base_coin], coin_image)
        num_matches /= ratio(base_coin_matches[base_coin])
        print(f"matches for {base_coin}: {num_matches}")
        if num_matches > max_matches and num_matches > threshold:
            result = base_coin
            max_matches = num_matches

    return result


def circle_edges(image):
    # Read the image
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (255, 255), 0)
    big_kernel = np.ones((11, 11), dtype=np.uint8)
    small_kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = cv2.dilate(blurred, big_kernel, iterations=1)
    edges = dilated - blurred
    _, edges = cv2.threshold(edges, 3, 255, cv2.THRESH_BINARY)
    edges = cv2.erode(edges, big_kernel, iterations=3)
    return edges


def find_coins(image, threshold=450, draw_circles_on_image=False):
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
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        30,
        param1=50,
        param2=27,
        minRadius=200,
        maxRadius=0,
    )
    circle_list = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        grouped_circles = {}
        for x, y, r in circles:
            if draw_circles_on_image and False:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
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
            circle_list.append([avg_x, avg_y, int(avg_r * 1.07)])

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


def find_best_matching_template(templates: dict, input_image: MatLike):
    best_match_distance = float('inf')
    best_match_index = -1
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors for input image
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    kp_input, des_input = sift.detectAndCompute(input_gray, None)
    
    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher_create()
    
    for key in templates.keys():
        # Detect keypoints and descriptors for template image
        template_gray = cv2.cvtColor(templates[key], cv2.COLOR_BGR2GRAY)
        kp_template, des_template = sift.detectAndCompute(template_gray, None)
        
        if des_template is None:
            continue
        
        # Match descriptors between input image and template
        matches = flann.knnMatch(des_template, des_input, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 0:
            # Calculate average distance of matched keypoints
            distance = np.mean([m.distance for m in good_matches])
            
            # Update best match if current match distance is lower
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_index = key
    
    return best_match_index


def main():
    masked_images = calculate_masked_images()
    for image in os.listdir("imgs"):
        target_image = cv2.imread(f"imgs/{image}")
        print(sum_coins(masked_images, target_image))

    # print(count_object_appearances(template_image, target_image))


if __name__ == "__main__":
    main()
