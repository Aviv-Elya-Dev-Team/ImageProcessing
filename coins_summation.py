import cv2
import os
from matplotlib import pyplot as plt
from coin_radius_finder import mask_coin, base_images, mask_circular_objects

# focus the images on the coins so that the SIFT algorithm only finds features inside
# the coins and not elsewhere in the image


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


def count_coin_appearances(template_image, target_image):

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
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Extract the location of keypoints in the target image
    target_keypoint_locations = [
        keypoints_target[match.trainIdx].pt for match in good_matches
    ]

    # Count the number of unique occurrences
    unique_locations = set(target_keypoint_locations)
    num_occurrences = len(unique_locations)

    # Draw matches
    result = cv2.drawMatches(
        template_image,
        keypoints_template,
        target_image,
        keypoints_target,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    show_image_plt(result)
    return num_occurrences


def sum_coin_values_sift(image_path, threshold=0.75):
    # read the input image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize sum of coin values
    total_value = 0

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Compute keypoints and descriptors for input image
    kp_input, des_input = sift.detectAndCompute(input_image, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors of input image with each template
    for value, template in base_images.items():
        # Compute keypoints and descriptors for template
        kp_template, des_template = sift.detectAndCompute(template, None)

        # Match descriptors
        matches = bf.knnMatch(des_template, des_input, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        # Add the value of each matched coin to the total
        total_value += value * len(good_matches)

    return total_value


def sum_coin_occurrences(masked_images, target_image):
    base_coins_values = [2, 5, 10, 50]
    base_images_num_matches = {
        2: count_coin_appearances(masked_images[2], masked_images[2]),
        5: count_coin_appearances(masked_images[5], masked_images[5]),
        10: count_coin_appearances(masked_images[10], masked_images[10]),
        50: count_coin_appearances(masked_images[50], masked_images[50]),
    }
    result = {2: 0, 5: 0, 10: 0, 50: 0}
    print(base_images_num_matches)
    for base_coin in base_coins_values:
        result[base_coin] = count_coin_appearances(masked_images[base_coin], target_image)

    print(result)


def main():
    masked_images = calculate_masked_images()

    template_image = masked_images[2]  # Provide the path to your input image here
    target_image = cv2.imread("imgs/62.jpg")

    sum_coin_occurrences(masked_images, target_image)

    # print(count_object_appearances(template_image, target_image))


if __name__ == "__main__":
    main()
