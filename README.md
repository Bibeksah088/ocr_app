# 🔬 OCR Research Lab

> A production-grade Optical Character Recognition system with real-time camera support, multi-mode preprocessing, batch processing, and detailed analytics.

---

## 📌 Project Overview

OCR Research Lab is a research-oriented application that extracts text from images using the **Tesseract 5.x OCR engine** combined with a customizable **OpenCV preprocessing pipeline**. It supports both static image upload and real-time camera input, making it suitable for document digitization, ID card scanning, printed text extraction, and live text recognition.

This project was built as a research tool to study the effect of various image preprocessing techniques on OCR accuracy.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Image Upload OCR** | JPG, PNG, BMP, TIFF, WEBP support |
| **Real-time Camera** | Snapshot mode + WebRTC live stream |
| **9 Preprocessing Modes** | Grayscale, Adaptive Threshold, OTSU, Denoise, Sharpen, Contrast, Deskew, Full Pipeline |
| **Bounding Box Visualization** | Color-coded by confidence (green/orange/red) |
| **Word-level Analysis** | Per-word confidence scores with sortable table |
| **Batch Processing** | Process multiple images, download combined report |
| **Mode Comparison** | Side-by-side accuracy comparison of all 9 modes |
| **Multi-language** | English, Hindi, French, German, Spanish, Portuguese, Italian |
| **PSM / OEM Control** | Full Tesseract page segmentation and engine control |
| **Export** | TXT, PDF (styled), Annotated image PNG, CSV |
| **Session Analytics** | Confidence trends, word counts, processing times |

---

## 🛠️ Installation

### 1. Install Tesseract OCR Engine

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
# For additional languages:
sudo apt-get install tesseract-ocr-hin tesseract-ocr-fra tesseract-ocr-deu
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # all language packs
```

**Windows:**
Download installer from https://github.com/UB-Mannheim/tesseract/wiki

### 2. Clone and Install Python Dependencies

```bash
git clone https://github.com/yourusername/ocr-research-lab
cd ocr-research-lab
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ocr_lab/
├── app.py                  # Main Streamlit application
├── config.py               # Global constants (languages, modes)
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── ocr_engine.py       # Tesseract wrapper (extract, confidence, boxes)
    ├── preprocessor.py     # OpenCV image preprocessing pipeline
    └── exporter.py         # TXT / PDF / CSV export
```

---

## 🔧 Preprocessing Pipeline Explained

| Mode | Description | Best For |
|---|---|---|
| None (Original) | Raw image, no processing | Clean, high-res images |
| Grayscale | Remove color channels | Most documents |
| Adaptive Threshold | Local thresholding per region | Uneven lighting |
| OTSU Threshold | Global optimal threshold | High-contrast text |
| Denoise | Non-local means denoising | Noisy/scanned images |
| Sharpen | Edge-enhancement kernel | Blurry images |
| Contrast Enhance | 2x contrast boost | Faded text |
| Deskew | Rotation correction | Tilted documents |
| **Full Pipeline** | Denoise → Grayscale → Deskew → Adaptive Threshold | **Most images (recommended)** |

---

## 📊 Architecture

```
Image Input (Upload / Camera)
        │
        ▼
ImagePreprocessor (OpenCV)
   └─ 9 configurable modes
        │
        ▼
OCREngine (pytesseract → Tesseract 5)
   ├─ extract_text()         → raw string
   ├─ extract_with_data()    → word positions + confidence
   ├─ get_confidence()       → mean confidence score
   ├─ draw_boxes()           → annotated PIL image
   └─ text_stats()           → chars / words / lines / numbers
        │
        ▼
Streamlit UI
   ├─ Tab 1: Image Upload
   ├─ Tab 2: Live Camera (Snapshot / WebRTC)
   ├─ Tab 3: Batch Processing
   ├─ Tab 4: Mode Comparison
   └─ Tab 5: Session Analytics
        │
        ▼
Exporter
   ├─ .txt  (with metadata header)
   ├─ .pdf  (styled, via fpdf2)
   └─ .csv  (session log)
```

---

## 🎛️ Configuration Options

| Setting | Options | Effect |
|---|---|---|
| Language | eng, hin, fra, deu, spa... | Tesseract dictionary used |
| Preprocessing | 9 modes | Image quality before OCR |
| PSM | 3 (Auto) → 10 (Single Char) | Page layout assumption |
| OEM | 0 (Legacy) → 3 (Best/LSTM) | Tesseract neural net mode |
| Confidence Threshold | 0–100 | Minimum confidence for box display |

---

## 📦 Tech Stack

- **Tesseract 5.x** — LSTM-based OCR engine
- **pytesseract** — Python wrapper for Tesseract
- **OpenCV 4.x** — Image preprocessing and annotation
- **Pillow** — Image I/O
- **Streamlit** — Web UI
- **streamlit-webrtc** — Real-time camera stream
- **fpdf2** — PDF generation
- **pandas** — Data analysis and export

---

## 🔮 Future Work

- [ ] GPU-accelerated preprocessing
- [ ] EasyOCR / PaddleOCR engine comparison
- [ ] REST API endpoint (FastAPI)
- [ ] Table/form structure detection
- [ ] Handwriting recognition mode
- [ ] Docker containerization

---

## 📄 License

MIT License — free to use, modify, and distribute.
