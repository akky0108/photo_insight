import cv2
import numpy as np

# Blurriness Evaluator
class BlurrinessEvaluator:
    def evaluate(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return 1 / (np.mean(sobel_magnitude) + 1e-6)