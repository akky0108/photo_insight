import cv2

# Sharpness Evaluator
class SharpnessEvaluator:
    def evaluate(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()