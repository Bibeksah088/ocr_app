VERSION = "1.0.0"

LANGS = {
    "English": "eng",
    "Hindi": "hin",
    "French": "fra",
    "German": "deu",
    "Spanish": "spa",
    "Portuguese": "por",
    "Italian": "ita",
}

PSM_MODES = {
    "Auto (Fully Automatic)": 3,
    "Single Column": 4,
    "Single Uniform Block": 6,
    "Single Text Line": 7,
    "Single Word": 8,
    "Single Character": 10,
}

OEM_MODES = {
    "Legacy Engine": 0,
    "LSTM Neural Net": 1,
    "Legacy + LSTM": 2,
    "Default (Best)": 3,
}

PREPROCESS_MODES = [
    "None (Original)",
    "Grayscale",
    "Adaptive Threshold",
    "OTSU Threshold",
    "Denoise",
    "Sharpen",
    "Contrast Enhance",
    "Deskew",
    "Full Pipeline (Recommended)",
]

SPEECH_LANGS = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "German": "de-DE",
    "French": "fr-FR",
    "Spanish": "es-ES",
}

