import cv2

# Noise Evaluator
class NoiseEvaluator:
    def evaluate(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image.std()
