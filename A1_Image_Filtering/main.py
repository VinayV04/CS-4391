import numpy as np
import cv2

source_image = cv2.imread("C:\\school github projects\\CS-4391\\A1_Image_Filtering\\lena_512.jpg",
                          cv2.IMREAD_GRAYSCALE).astype(np.float32)

def save_and_display_image(img, file_name):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(file_name, img_uint8)
    cv2.imshow(file_name, img_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_filter(image, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    output = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image,
                          ((pad_h, pad_h), (pad_w, pad_w)),
                          mode='constant', constant_values=0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(region * kernel)

    return output

# 7x7 Box Blur Filter
def box_blur(image, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return apply_filter(image, kernel)

box_blur_image = box_blur(source_image)
save_and_display_image(box_blur_image, 'box_blur_7x7.jpg')

# 15x15 Gaussian Blur Filter
def generate_gaussian_kernel(dim, sigma=1.0):
    axis = np.linspace(-(dim // 2), dim // 2, dim)
    x_grid, y_grid = np.meshgrid(axis, axis)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2)).astype(np.float32)
    return kernel / np.sum(kernel)

def apply_gaussian_blur(img, size=15, sigma=2.0):
    gaussian_kernel = generate_gaussian_kernel(size, sigma)
    return apply_filter(img, gaussian_kernel)

gaussian_blur_result = apply_gaussian_blur(source_image)
save_and_display_image(gaussian_blur_result, 'gaussian_blur_15x15.jpg')

# 15x15 Linear Motion Blur Filter
def linear_motion_blur(img, size=15):
    motion_kernel = np.zeros((size, size), dtype=np.float32)
    np.fill_diagonal(motion_kernel, 1.0 / size)
    return apply_filter(img, motion_kernel)

motion_blur_result = linear_motion_blur(source_image)
save_and_display_image(motion_blur_result, 'motion_blur_15x15.jpg')

# 3x3 Laplacian Edge Enhancement Filter
def laplacian_edge_enhance(img):
    laplacian_kernel = np.array([[0, -1,  0],
                                 [-1,  5, -1],
                                 [0, -1,  0]], dtype=np.float32)
    return apply_filter(img, laplacian_kernel)

laplacian_enhance_result = laplacian_edge_enhance(source_image)
save_and_display_image(laplacian_enhance_result, 'laplacian_3x3.jpg')

# Canny Edge Detection Filter
def canny_edge_detection(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1,
                         -2,
                         -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gradient_x = apply_filter(img, sobel_x)
    gradient_y = apply_filter(img, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    return gradient_magnitude.astype(np.uint8)

canny_edge_result = canny_edge_detection(source_image)
save_and_display_image(canny_edge_result, 'canny_edge.jpg')
