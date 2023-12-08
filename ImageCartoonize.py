import cv2
import numpy as np
from tkinter import Tk, filedialog

# Suppressing main window
root = Tk()
root.withdraw()

photo = filedialog.askopenfilename()
img = cv2.imread(photo)

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

# Define the parameters for edge detection
line_width = 9      #Adjust the value of as needed
blur_value = 7
totalColors= 5

# Get edges using the defined function
edges = edge_detection(img, line_width, blur_value)

# Cartoonize with bilateral filter
color = cv2.bilateralFilter(img, 5, 255, 255)
cartoon_with_edges = cv2.bitwise_and(color, color, mask=edges)

# Stylization effect

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Example usage

cartoon= color_quantisation(cartoon_with_edges, 9)  # Adjust the value of k as needed = color_quantisation(img, 8)  # Adjust the value of k as needed
cv2.imshow("Original Image",img)
cv2.imshow("cartoon_with_edges",cartoon_with_edges)
cv2.imshow("Quantized Image", cartoon)

cv2.imwrite("cartoon.jpg", cartoon)


cv2.waitKey(0)
cv2.destroyAllWindows()
