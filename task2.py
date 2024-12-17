import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to capture image from camera
def capture_image(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture an image.")
        return None
    return frame


# Function to convert to grayscale
def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Function to threshold an image into black and white
def threshold_bw(image, threshold=127):
    gray = grayscale_image(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


# Function to threshold into 16 gray colors (8 regions)
def threshold_16_gray(image):
    gray = grayscale_image(image)
    step = 32  # Divide 256 into 8 regions (32 each)
    quantized = (gray // step) * step
    return quantized


# Apply Sobel filter and Canny edge detector
def sobel_canny(image):
    gray = grayscale_image(image)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    canny = cv2.Canny(gray, 50, 150)
    return sobel, canny


# Apply Gaussian filter to remove noise
def gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


# Sharpen an image using a sharpening kernel
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# Convert RGB to BGR
def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# Display images in a 2x4 grid
def show_images_grid(images, titles, n=2, m=4):
    plt.figure(figsize=(15, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.subplot(n, m, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    source = 0  # Use 0 or 1 depending on camera
    captured_image = capture_image(source)

    if captured_image is not None:
        # a. Grayscale
        gray_image = grayscale_image(captured_image)

        # b. Thresholding
        bw_image = threshold_bw(captured_image)
        gray16_image = threshold_16_gray(captured_image)

        # c. Sobel filter and Canny edge detector
        sobel_image, canny_image = sobel_canny(captured_image)

        # d. Gaussian filter
        blurred_image = gaussian_blur(gray_image)

        # e. Sharpen the blurred image
        sharpened_image = sharpen_image(blurred_image)

        # f. RGB to BGR conversion
        bgr_image = rgb_to_bgr(captured_image)

        # Prepare images and titles for the grid
        images = [
            captured_image, gray_image, bw_image, gray16_image,
            sobel_image, canny_image, blurred_image, sharpened_image
        ]
        titles = [
            "Original", "Grayscale", "Black & White", "16 Gray Colors",
            "Sobel Filter", "Canny Edge", "Gaussian Blur", "Sharpened Image"
        ]

        # Display the grid
        show_images_grid(images, titles, n=2, m=4)

        print("Default color channel in OpenCV is BGR.")
