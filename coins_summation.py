import cv2
from matplotlib import pyplot as plt

# focus the images on the coins so that the SIFT algorithm only finds features inside
# the coins and not elsewhere in the image
base_images = {
    2: cv2.imread("imgs/2.jpg")[1800:3400, 900:2500],
    5: cv2.imread("imgs/5.jpg")[2025:2925, 1175:2075],
    10: cv2.imread("imgs/10.jpg")[2160:3060, 1310:2210],
    50: cv2.imread("imgs/50.jpg")[2235:3320, 865:2000],
}

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


def count_object_appearances(image_path, coin_val):
    thresholds = {2: 0.65, 5: 0.75, 10: 0.75, 50: 0.75}
    # Read images
    img1 = base_images[coin_val]
    img1 = clean_image(img1)
    img2 = cv2.imread(image_path)
    img2 = clean_image(img2)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    kp = sift.detect(img1, None)
    image = cv2.drawKeypoints(img1, kp, img1)

    show_image_plt(image)

    # Initialize brute force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < thresholds[coin_val] * n.distance:
            good_matches.append(m)

    # Count the number of good matches
    return len(good_matches)


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


def main():
    # Example usage:
    image_path = "imgs/2.jpg"  # Provide the path to your input image here
    """
    total = sum_coin_values_sift(image_path, 0.53)
    print("Total value of coins in the image:", total)
    """
    print(count_object_appearances(image_path, 2))

if __name__ == "__main__":
    main()