from PIL import Image
import numpy as np

def determine_flag_sobel(image_path):
    try:
        # Load and resize the input image to a manageable size (200x100)
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((200, 100))
        img_data = np.array(image, dtype='int32')

        # Define Sobel filter kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Initialize gradient arrays
        gradient_x = np.zeros_like(img_data)
        gradient_y = np.zeros_like(img_data)

        # Apply Sobel filters to detect edges
        for i in range(1, img_data.shape[0] - 1):
            for j in range(1, img_data.shape[1] - 1):
                region = img_data[i - 1:i + 2, j - 1:j + 2]
                gradient_x[i, j] = np.sum(region * sobel_x)
                gradient_y[i, j] = np.sum(region * sobel_y)

        # Combine X and Y gradients to get the Sobel magnitude
        sobel_combined = np.hypot(gradient_x, gradient_y)
        sobel_combined = np.clip(sobel_combined, 0, 255).astype('uint8')

        # Analyze the Sobel data
        upper_half = np.mean(sobel_combined[:50, :])
        lower_half = np.mean(sobel_combined[50:, :])


        if upper_half > lower_half:
            print("Detected the Indonesia flag.")
        else:
            print("Detected Poland flag.")

    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")

    finally:
        print(f"Processing completed for {image_path}")


file_path = 'poland.jpeg'

determine_flag_sobel(file_path)
