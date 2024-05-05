import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from coin_radius_finder import mask_coin, base_images
import sys

# typing
import numpy.typing

MatLike = numpy.typing.NDArray[numpy.uint8]


def show_image(image, title="Figure"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.gcf().canvas.manager.set_window_title(sys.argv[1].split('/')[-1])
    plt.show()


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
            2: cv2.cvtColor(mask_coin(base_images[2]), cv2.COLOR_RGB2GRAY),
            5: cv2.cvtColor(mask_coin(base_images[5]), cv2.COLOR_RGB2GRAY),
            10: cv2.cvtColor(mask_coin(base_images[10]), cv2.COLOR_RGB2GRAY),
            50: cv2.cvtColor(mask_coin(base_images[50]), cv2.COLOR_RGB2GRAY),
        }

    for key in masked_images.keys():
        cv2.imwrite(f"templates/{str(key)}.png", masked_images[key])
    print("done calculating masks")
    return masked_images


def count_matches(
    template_image: MatLike, target_image: MatLike, threshold=0.8, draw_matches=False
):
    """
    counts how many good matches are in the target_image from the template_image

    Args:
        template_image (MatLike): input image to extract matches from and search in target_image
        target_image (MatLike): image to find matches in
        threshold (float, optional): filters out matches too close to each other. Defaults to 0.8.

    Returns:
        int: number of good matches (within the threshold) found
    """
    # use SIFT to find keypoints inside target_image from template_image
    sift = cv2.SIFT_create()
    keypoints_template, descriptors_template = sift.detectAndCompute(
        template_image, None
    )
    keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)

    # use knn to find the best matches
    # (keep around 80% of them. adjustable using the threshold)
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors_template, descriptors_target, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    if draw_matches:
        matched_img = cv2.drawMatches(
            template_image,
            keypoints_template,
            target_image,
            keypoints_target,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        show_image(matched_img)
    return len(good_matches)


def sum_coins(masked_images, target_image):
    base_coin_matches = {
        2: count_matches(masked_images[2], masked_images[2]),
        5: count_matches(masked_images[5], masked_images[5]),
        10: count_matches(masked_images[10], masked_images[10]),
        50: count_matches(masked_images[50], masked_images[50]),
    }
    result = 0

    # extract just the coins from the target image
    target_coins_circles = find_coins(target_image)
    target_coins_images = [
        extract_coin_from_image(target_image, (coin[0], coin[1]), coin[2])
        for coin in target_coins_circles
    ]

    # classify each coin from the target image and add to the sum
    for i, target_coin_image in enumerate(target_coins_images):
        predicted_coin = classify_from_image(
            masked_images, base_coin_matches, target_coin_image
        )
        draw_prediction_on_target(target_image, target_coins_circles[i], predicted_coin)
        result += predicted_coin

    return result


def draw_prediction_on_target(target_image, target_coin_circle, predicted_coin):
    # draw the predicted coin above the target coin in the target image
    x, y, radius = target_coin_circle

    # a rectangle around the coin
    cv2.rectangle(
        target_image,
        (x - radius, y - radius),
        (x + radius, y + radius),
        (255, 0, 0),
        10,
    )

    # background for the text
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 12
    text_size, _ = cv2.getTextSize(f"val: {predicted_coin}", font, fontScale, 20)
    cv2.rectangle(
        target_image,
        (int(x - radius), int(y - radius - text_size[1] - 20)),
        (int(x - radius + text_size[0]), int(y - radius)),
        (255, 255, 255),
        cv2.FILLED,
    )

    # the text
    cv2.putText(
        target_image,
        f"val: {predicted_coin}",
        (int(x - radius), int(y - radius)),
        font,
        fontScale,
        color=(0, 200, 0),
        thickness=20,
        lineType=cv2.FILLED,
    )


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

    def ratio(base_coin):
        if base_coin == 50:
            return 0.2006299344848081
        if base_coin == 10:
            return 0.6981966664335424
        if base_coin == 5:
            return 0.11829741638410693
        if base_coin == 2:
            return 0.5235495972124882

    result = 0
    max_matches = 0
    for base_coin in masked_images.keys():
        num_matches = count_matches(masked_images[base_coin], coin_image)
        after_ratio = num_matches * ratio(base_coin)
        num_matches = after_ratio
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
    circles = cv2.HoughCircles(
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
            if draw_circles_on_image:
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
            if len(circle_group) < 2:
                continue
            group_x, group_y, group_r = zip(*circle_group)
            avg_x = int(np.mean(group_x))
            avg_y = int(np.mean(group_y))
            avg_r = int(np.mean(group_r))
            circle_list.append([avg_x, avg_y, int(avg_r * 1.07)])

        # draw the average circles on the original image if the user wants
        if draw_circles_on_image:
            for avg_x, avg_y, avg_r in circle_list:
                cv2.circle(image, (avg_x, avg_y), avg_r, (255, 0, 0), 10)

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

    return cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)


def main(args):
    if len(args) != 1:
        print(
            "format: python coins_summation.py <FILENAME>\nexample: python coins_summation.py 62.jpg"
        )
        return
    image = args[0]
    masked_images = calculate_masked_images()
    target_image = cv2.imread(f"{image}")

    if target_image is None:
        print("image not found\nformat: python coins_summation.py <FILENAME>")
        return
    
    print("Please wait, this might take a while...")
    result = sum_coins(masked_images, target_image)
    print("Done")

    show_image(target_image, f"sum: {result}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
