from datetime import datetime
import pandas as pd
import io


class Exporter:

    @staticmethod
    def to_txt(text, meta=None):
        out = ""
        if meta:
            out += "=" * 40 + "\n"
            out += "  OCR RESEARCH LAB — EXTRACTION REPORT\n"
            out += "=" * 40 + "\n"
            out += f"Date       : {meta.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n"
            out += f"Confidence : {meta.get('confidence', 'N/A')}%\n"
            out += f"Language   : {meta.get('lang', 'English')}\n"
            out += f"Preprocess : {meta.get('preprocess', 'None')}\n"
            out += f"Words      : {meta.get('words', 'N/A')}\n"
            out += "=" * 40 + "\n\n"
        out += text
        return out.encode("utf-8")

    @staticmethod
    def to_pdf_bytes(text, meta=None):
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()

            if meta:
                pdf.set_fill_color(15, 15, 35)
                pdf.rect(0, 0, 210, 40, "F")
                pdf.set_text_color(167, 139, 250)
                pdf.set_font("Helvetica", "B", 18)
                pdf.set_y(12)
                pdf.cell(0, 10, "OCR Research Lab", ln=True, align="C")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(156, 163, 175)
                pdf.cell(0, 6, "Extraction Report", ln=True, align="C")
                pdf.ln(10)

                pdf.set_text_color(100, 100, 120)
                pdf.set_font("Helvetica", "", 9)
                fields = [
                    ("Date", meta.get("date", "")),
                    ("Confidence", f"{meta.get('confidence', 'N/A')}%"),
                    ("Language", meta.get("lang", "English")),
                    ("Preprocessing", meta.get("preprocess", "None")),
                    ("Words Detected", str(meta.get("words", "N/A"))),
                ]
                for label, val in fields:
                    pdf.set_text_color(120, 120, 140)
                    pdf.cell(45, 7, f"{label}:", border=0)
                    pdf.set_text_color(50, 50, 70)
                    pdf.cell(0, 7, val, ln=True)

                pdf.set_draw_color(167, 139, 250)
                pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
                pdf.ln(8)

            pdf.set_text_color(40, 40, 60)
            pdf.set_font("Courier", "", 11)
            for line in text.split("\n"):
                try:
                    pdf.multi_cell(0, 6, line)
                except Exception:
                    safe = line.encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe)

            return bytes(pdf.output())
        except Exception:
            return None

    @staticmethod
    def to_csv(records):
        if not records:
            return b""
        return pd.DataFrame(records).to_csv(index=False).encode("utf-8")
