import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImagePreprocessor:

    @staticmethod
    def _to_cv2(img):
        if isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _to_pil(img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                return Image.fromarray(img)
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    @staticmethod
    def none_mode(img):
        return ImagePreprocessor._to_cv2(img)

    @staticmethod
    def grayscale(img):
        cv = ImagePreprocessor._to_cv2(img)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_threshold(img):
        cv = ImagePreprocessor._to_cv2(img)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def otsu_threshold(img):
        cv = ImagePreprocessor._to_cv2(img)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, t = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def denoise(img):
        cv = ImagePreprocessor._to_cv2(img)
        return cv2.fastNlMeansDenoisingColored(cv, None, 10, 10, 7, 21)

    @staticmethod
    def sharpen(img):
        cv = ImagePreprocessor._to_cv2(img)
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(cv, -1, k)

    @staticmethod
    def contrast_enhance(img):
        pil = ImagePreprocessor._to_pil(img) if not isinstance(img, Image.Image) else img
        enhanced = ImageEnhance.Contrast(pil).enhance(2.0)
        return ImagePreprocessor._to_cv2(enhanced)

    @staticmethod
    def deskew(img):
        cv = ImagePreprocessor._to_cv2(img)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inv > 0))
        if len(coords) == 0:
            return cv
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        h, w = cv.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def full_pipeline(img):
        cv = ImagePreprocessor._to_cv2(img)
        h, w = cv.shape[:2]
        if w < 1000:
            sc = 1000 / w
            cv = cv2.resize(cv, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        cv = cv2.fastNlMeansDenoisingColored(cv, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 0.5:
                h2, w2 = gray.shape
                M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kern)
        return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

   _MAP = {
    "None (Original)": none_mode,
    "Grayscale": grayscale,
    "Adaptive Threshold": adaptive_threshold,
    "OTSU Threshold": otsu_threshold,
    "Denoise": denoise,
    "Sharpen": sharpen,
    "Contrast Enhance": contrast_enhance,
    "Deskew": deskew,
    "Full Pipeline (Recommended)": full_pipeline,
}


   @classmethod
def process(cls, img, mode):
    fn = cls._MAP.get(mode, cls.none_mode)
    result = fn(img)
    return cls._to_pil(result)
