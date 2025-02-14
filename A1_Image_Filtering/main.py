import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# load image and convert to grayscale
source_image = img = cv2.imread("C:\\school github projects\\CS-4391\\A1_Image_Filtering\\lena_512.jpg", cv2.IMREAD_GRAYSCALE)

def save_and_display_image(img, file_name):
    cv2.imwrite(file_name, img)
    plt.imshow(img, cmap='gray')
    plt.title(file_name)
    plt.axis('off')
    plt.show()

# 7x7 box blur filter
def box_blur(image, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return convolve2d(image, kernel, mode='same', boundary='symm').astype(np.uint8)


box_blur_image = box_blur(source_image)
save_and_display_image(box_blur_image, 'box_blur_7x7.jpg')

# 15x15 gaussian smoothing filter
def generate_gaussian_kernel(dim, sigma=1.0):
    axis = np.linspace(-(dim // 2), dim // 2, dim)
    x_grid, y_grid = np.meshgrid(axis, axis)
    kernel = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def apply_gaussian_blur(img, size=15, sigma=2.0):
    gaussian_kernel = generate_gaussian_kernel(size, sigma)
    return convolve2d(img, gaussian_kernel, mode='same', boundary='symm').astype(np.uint8)

gaussian_blur_result = apply_gaussian_blur(source_image)
save_and_display_image(gaussian_blur_result, 'gaussian_blur_15x15.jpg')

# 15x15 linear motion blur filter
def linear_motion_blur(img, size=15):
    motion_kernel = np.zeros((size, size))
    np.fill_diagonal(motion_kernel, 1.0 / size)
    return convolve2d(img, motion_kernel, mode='same', boundary='symm').astype(np.uint8)

motion_blur_result = linear_motion_blur(source_image)
save_and_display_image(motion_blur_result, 'motion_blur_15x15.jpg')

# 3x3 laplacian edge enhancement filter
def laplacian_edge_enhance(img):
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    return convolve2d(img, laplacian_kernel, mode='same', boundary='symm').astype(np.uint8)

laplacian_enhance_result = laplacian_edge_enhance(source_image)
save_and_display_image(laplacian_enhance_result, 'laplacian_3x3.jpg')

# edge detection using canny filter
def canny_edge_detection(img, low_thresh=50, high_thresh=150):

    smoothed_image = apply_gaussian_blur(img, size=5, sigma=1.0)

    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolve2d(smoothed_image, sobel_filter_x, mode='same', boundary='symm')
    gradient_y = convolve2d(smoothed_image, sobel_filter_y, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2).astype(np.uint8)
    gradient_angle = np.arctan2(gradient_y, gradient_x)

    suppressed_edges = np.zeros_like(gradient_magnitude)
    angle_deg = gradient_angle * 180.0 / np.pi
    angle_deg[angle_deg < 0] += 180

    for x in range(1, gradient_magnitude.shape[0] - 1):
        for y in range(1, gradient_magnitude.shape[1] - 1):
            neighbor1, neighbor2 = 255, 255

            if (0 <= angle_deg[x, y] < 22.5) or (157.5 <= angle_deg[x, y] <= 180):
                neighbor1 = gradient_magnitude[x, y + 1]
                neighbor2 = gradient_magnitude[x, y - 1]
            elif (22.5 <= angle_deg[x, y] < 67.5):
                neighbor1 = gradient_magnitude[x + 1, y - 1]
                neighbor2 = gradient_magnitude[x - 1, y + 1]
            elif (67.5 <= angle_deg[x, y] < 112.5):
                neighbor1 = gradient_magnitude[x + 1, y]
                neighbor2 = gradient_magnitude[x - 1, y]
            elif (112.5 <= angle_deg[x, y] < 157.5):
                neighbor1 = gradient_magnitude[x - 1, y - 1]
                neighbor2 = gradient_magnitude[x + 1, y + 1]

            if gradient_magnitude[x, y] >= neighbor1 and gradient_magnitude[x, y] >= neighbor2:
                suppressed_edges[x, y] = gradient_magnitude[x, y]
            else:
                suppressed_edges[x, y] = 0

    strong_intensity = 255
    weak_intensity = 50

    strong_x, strong_y = np.where(suppressed_edges >= high_thresh)
    weak_x, weak_y = np.where((suppressed_edges <= high_thresh) & (suppressed_edges >= low_thresh))

    final_edges = np.zeros_like(suppressed_edges)
    final_edges[strong_x, strong_y] = strong_intensity
    final_edges[weak_x, weak_y] = weak_intensity

    for x in range(1, final_edges.shape[0] - 1):
        for y in range(1, final_edges.shape[1] - 1):
            if final_edges[x, y] == weak_intensity:
                if (strong_intensity in final_edges[x - 1:x + 2, y - 1:y + 2]):
                    final_edges[x, y] = strong_intensity
                else:
                    final_edges[x, y] = 0

    return final_edges

canny_edge_result = canny_edge_detection(source_image)
save_and_display_image(canny_edge_result, 'canny_edge.jpg')