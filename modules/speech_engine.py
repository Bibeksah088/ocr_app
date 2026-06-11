import io
import speech_recognition as sr
from pydub import AudioSegment


class SpeechEngine:

    LANGS = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "German": "de-DE",
        "French": "fr-FR",
        "Spanish": "es-ES",
    }

    @staticmethod
    def transcribe_bytes(audio_bytes, lang="en-US"):
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.pause_threshold = 0.8

        try:
            webm_io = io.BytesIO(audio_bytes)
            wav_io = io.BytesIO()

            AudioSegment.from_file(
                webm_io,
                format="webm"
            ).export(
                wav_io,
                format="wav"
            )

            wav_io.seek(0)

            with sr.AudioFile(wav_io) as src:
                r.adjust_for_ambient_noise(src, duration=0.2)
                audio = r.record(src)

            text = r.recognize_google(audio, language=lang)

            return text.strip(), None

        except sr.UnknownValueError:
            return "", "Could not understand audio. Please speak clearly and try again."

        except sr.RequestError as e:
            return "", f"Google API unavailable: {e}. Check your internet connection."

        except Exception as e:
            return "", f"Audio processing error: {str(e)}"

    @staticmethod
    def transcribe_file(file_obj, lang="en-US"):
        r = sr.Recognizer()

        try:
            with sr.AudioFile(file_obj) as src:
                audio = r.record(src)

            text = r.recognize_google(audio, language=lang)

            return text.strip(), None

        except sr.UnknownValueError:
            return "", "Could not understand audio."

        except sr.RequestError as e:
            return "", f"Google API error: {e}"

        except Exception as e:
            return "", f"Error: {str(e)}"