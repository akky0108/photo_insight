import cv2
import numpy as np
import pywt

# Wavelet Sharpness Evaluator
class WaveletSharpnessEvaluator:
    def evaluate(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coeffs2 = pywt.dwt2(gray_image, 'haar')
        LL, (LH, HL, HH) = coeffs2
        return np.mean(np.abs(HH))