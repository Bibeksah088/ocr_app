import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import Image
import time
import re


class OCREngine:

    @staticmethod
    def _cfg(psm, oem):
        return f"--psm {psm} --oem {oem}"

    @staticmethod
    def extract_text(img, lang="eng", psm=3, oem=3):
        t0 = time.time()
        txt = pytesseract.image_to_string(img, lang=lang, config=OCREngine._cfg(psm, oem))
        return txt.strip(), time.time() - t0

    @staticmethod
    def extract_with_data(img, lang="eng", psm=3, oem=3):
        t0 = time.time()
        data = pytesseract.image_to_data(img, lang=lang, config=OCREngine._cfg(psm, oem), output_type=Output.DICT)
        return data, time.time() - t0

    @staticmethod
    def get_confidence(data):
        vals = [int(c) for c in data["conf"] if int(c) != -1]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    @staticmethod
    def draw_boxes(img, data, thresh=60):
        if isinstance(img, Image.Image):
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            cv_img = img.copy()
        for i in range(len(data["level"])):
            cf = int(data["conf"][i])
            if cf < thresh or cf == -1:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if w == 0 or h == 0:
                continue
            col = (0, 220, 120) if cf >= 80 else ((0, 165, 255) if cf >= 60 else (0, 80, 255))
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), col, 2)
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_word_stats(data, thresh=0):
        out = []
        for i, word in enumerate(data["text"]):
            cf = int(data["conf"][i])
            if word.strip() and cf > thresh:
                out.append({"word": word, "confidence": cf, "x": data["left"][i], "y": data["top"][i]})
        return out

    @staticmethod
    def text_stats(text):
        return {
            "characters": len(text),
            "words": len(text.split()),
            "lines": len([l for l in text.split("\n") if l.strip()]),
            "numbers": len(re.findall(r"\d+", text)),
        }
