import cv2

# Contrast Evaluator
class ContrastEvaluator:
    def evaluate(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image.max() - gray_image.min()
