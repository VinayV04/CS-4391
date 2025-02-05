import cv2 as cv

img = cv.imread("C:\\school github projects\\CS-4391\\A0_OpenCV_Setup\\lena_256.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window
