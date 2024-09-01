import imageio
import cv2
import numpy as np
import pywt

class PortraitQualityEvaluator:
    def __init__(self, rgb_image, weights=None):
        self.rgb_image = np.array(rgb_image)
        if weights is None:
            self.weights = {
                'sharpness': 0.2,
                'contrast': 0.2,
                'noise': 0.2,
                'wavelet_sharpness': 0.1,
                'face': 0.2,
                'blurriness': 0.1
            }
        else:
            self.weights = weights

    def calculate_sharpness(self, image=None):
        if image is None:
            image = self.rgb_image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness
    
    def calculate_contrast(self, image=None):
        if image is None:
            image = self.rgb_image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        contrast = gray_image.max() - gray_image.min()
        return contrast
    
    def calculate_noise(self, image=None):
        if image is None:
            image = self.rgb_image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise = gray_image.std()
        return noise
    
    def calculate_wavelet_sharpness(self, image=None):
        if image is None:
            image = self.rgb_image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coeffs2 = pywt.dwt2(gray_image, 'haar')
        LL, (LH, HL, HH) = coeffs2
        wavelet_sharpness = np.mean(np.abs(HH))
        return wavelet_sharpness
    
    def calculate_blurriness(self, image=None):
        if image is None:
            image = self.rgb_image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        blurriness = 1 / (np.mean(sobel_magnitude) + 1e-6)  # 値が大きいほどピンボケしている
        return blurriness

    def detect_face_and_evaluate(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_detected = len(faces) > 0
        
        if not face_detected:
            face_evaluation = {
                'sharpness': 0,
                'contrast': 0,
                'noise_level': 0,
                'wavelet_sharpness': 0,
                'blurriness': 0
            }
            face_weight = 0
            return face_evaluation, face_weight, face_detected
        
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_region = self.rgb_image[y:y+h, x:x+w]
        
        face_evaluation = {
            'sharpness': self.calculate_sharpness(face_region),
            'contrast': self.calculate_contrast(face_region),
            'noise_level': self.calculate_noise(face_region),
            'wavelet_sharpness': self.calculate_wavelet_sharpness(face_region),
            'blurriness': self.calculate_blurriness(face_region)
        }
        
        face_score = (
            face_evaluation['sharpness'] * self.weights['sharpness'] +
            face_evaluation['contrast'] * self.weights['contrast'] +
            face_evaluation['noise_level'] * self.weights['noise'] +
            face_evaluation['wavelet_sharpness'] * self.weights['wavelet_sharpness'] +
            face_evaluation['blurriness'] * self.weights['blurriness']
        )
        
        face_area = w * h
        image_area = self.rgb_image.shape[0] * self.rgb_image.shape[1]
        face_weight = (face_area / image_area) * self.weights['face']
        face_weight = min(face_weight, 1.0)
        
        return face_evaluation, face_weight, face_detected

    def evaluate(self):
        print("Evaluating image...")
        
        sharpness_score = self.calculate_sharpness()
        contrast_score = self.calculate_contrast()
        noise_score = self.calculate_noise()
        wavelet_sharpness_score = self.calculate_wavelet_sharpness()
        blurriness_score = self.calculate_blurriness()
        
        face_evaluation, face_weight, face_detected = self.detect_face_and_evaluate()
        
        if face_detected:
            weighted_scores = (
                sharpness_score * self.weights['sharpness'] +
                contrast_score * self.weights['contrast'] +
                noise_score * self.weights['noise'] +
                wavelet_sharpness_score * self.weights['wavelet_sharpness'] +
                blurriness_score * self.weights['blurriness'] +
                face_weight * sum(face_evaluation.values())
            )
        else:
            print("Using overall metrics since no face detected")
            weighted_scores = (
                sharpness_score * self.weights['sharpness'] +
                contrast_score * self.weights['contrast'] +
                noise_score * self.weights['noise'] +
                wavelet_sharpness_score * self.weights['wavelet_sharpness'] +
                blurriness_score * self.weights['blurriness']
            )
        
        overall_score = round(weighted_scores, 2)
        
        return {
            'overall_score': overall_score,
            'sharpness_score': sharpness_score,
            'contrast_score': contrast_score,
            'noise_score': noise_score,
            'wavelet_sharpness_score': wavelet_sharpness_score,
            'blurriness_score': blurriness_score,
            'face_evaluation': face_evaluation,
            'face_detected': face_detected
        }
    
if __name__ == "__main__":
    rgb_image = imageio.imread('portrait.jpg')
    custom_weights = {
        'sharpness': 0.2,
        'contrast': 0.2,
        'noise': 0.2,
        'wavelet_sharpness': 0.1,
        'face': 0.2,
        'blurriness': 0.1
    }
    evaluator = PortraitQualityEvaluator(rgb_image, weights=custom_weights)
    evaluation_results = evaluator.evaluate()

    print(evaluation_results)
