import cv2
import matplotlib.pyplot as plt

def apply_high_pass_filter(image):

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    high_pass = cv2.magnitude(sobelx, sobely)
    high_pass = cv2.convertScaleAbs(high_pass)
    return high_pass

def apply_low_pass_filter(image):

    low_pass = cv2.GaussianBlur(image, (7, 7), 0)
    return low_pass

def combine_filters(high_pass, low_pass):
    if high_pass.shape != low_pass.shape:
        low_pass = cv2.resize(low_pass, (high_pass.shape[1], high_pass.shape[0]))
    combined = cv2.addWeighted(high_pass, 0.5, low_pass, 0.5, 0)
    return combined


file_path = 'task3a.jpeg'
file_path0 = 'task3b.jpeg'

image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
image0 = cv2.imread(file_path0, cv2.IMREAD_GRAYSCALE)


if image is None:
    print("Image not found. Check the path.")
else:

    high_pass_image = apply_high_pass_filter(image)
    low_pass_image = apply_low_pass_filter(image0)
    combined_image = combine_filters(high_pass_image, low_pass_image)


    plt.figure(figsize=(15, 10))

    images = [
        (image, 'task3a.jpeg'),
        (high_pass_image, 'High-Pass Filter'),
        (combined_image, 'Combined Image'),
        (image0, 'task3b.jpeg'),
        (low_pass_image, 'Low-Pass Filter'),
    ]

    for idx, (img, title) in enumerate(images):
        plt.subplot(2, 3, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.show()
