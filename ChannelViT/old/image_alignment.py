import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def align_images(base_image, image_to_align):
    """Align image_to_align to base_image using feature matching and homography."""
    # Convert images to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(base_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(align_gray, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = base_image.shape
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))

    return aligned_image

def plot_aligned_images(images):
    """Plot aligned images in a grid."""

    base_image = images[0]
    aligned_images = [base_image]

    for image in images[1:]:
        aligned_image = align_images(base_image, image)
        aligned_images.append(aligned_image)


    fig, axes = plt.subplots(5, 5, figsize=(5 * 3, 5 * 3))
    axes = axes.flatten()

    for ax, image in zip(axes, aligned_images[:25]):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('aligned_images.png')

def find_optimal_grid(n):
    """Return the optimal grid dimensions for plotting n images."""
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
    return n, 1