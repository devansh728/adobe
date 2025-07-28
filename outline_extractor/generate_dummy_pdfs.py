from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def make_pdf(path, title, authors, sections):
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 60
    # Title
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width/2, y, title)
    y -= 40
    # Authors
    c.setFont("Helvetica", 12)
    for author in authors:
        c.drawCentredString(width/2, y, author)
        y -= 18
    y -= 10
    # Email/affiliation
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width/2, y, "author@email.com, University of Example")
    y -= 30
    # Sections
    for sec in sections:
        if y < 100:
            c.showPage()
            y = height - 60
        # H1
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, y, sec['h1'])
        y -= 24
        # H2
        for h2 in sec.get('h2', []):
            c.setFont("Helvetica-Bold", 13)
            c.drawString(80, y, h2['h2'])
            y -= 20
            # H3
            for h3 in h2.get('h3', []):
                c.setFont("Helvetica-Bold", 11)
                c.drawString(100, y, h3)
                y -= 16
            # Body
            c.setFont("Helvetica", 10)
            for _ in range(3):
                c.drawString(110, y, "This is some body text under the heading. It is just for training.")
                y -= 14
        y -= 10
    c.save()

def main():
    outdir = os.path.dirname(__file__)
    # PDF 1
    make_pdf(
        os.path.join(outdir, "sample1.pdf"),
        "Deep Learning for PDF Outlines",
        ["Alice Smith", "Bob Lee"],
        [
            {"h1": "Abstract", "h2": []},
            {"h1": "Introduction", "h2": [
                {"h2": "Motivation", "h3": ["Background", "Prior Work"]},
                {"h2": "Contributions", "h3": []}
            ]},
            {"h1": "Method", "h2": [
                {"h2": "Feature Extraction", "h3": ["Font Features", "Position Features"]},
                {"h2": "Model Training", "h3": []}
            ]},
            {"h1": "Results", "h2": [
                {"h2": "Experiments", "h3": []},
                {"h2": "Discussion", "h3": []}
            ]},
            {"h1": "Conclusion", "h2": []},
        ]
    )
    # PDF 2
    make_pdf(
        os.path.join(outdir, "sample2.pdf"),
        "A Survey on Multilingual Heading Detection",
        ["Carol Zhang", "David Kim"],
        [
            {"h1": "Resumen", "h2": []},
            {"h1": "Introducción", "h2": [
                {"h2": "Motivación", "h3": []},
                {"h2": "Contribuciones", "h3": []}
            ]},
            {"h1": "Método", "h2": [
                {"h2": "Extracción de Características", "h3": []},
                {"h2": "Entrenamiento del Modelo", "h3": []}
            ]},
            {"h1": "Resultados", "h2": []},
            {"h1": "Conclusión", "h2": []},
        ]
    )
    print("Dummy PDFs generated: sample1.pdf, sample2.pdf")

if __name__ == "__main__":
    main() 