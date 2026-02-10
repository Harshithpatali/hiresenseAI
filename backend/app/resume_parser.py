import pdfplumber
from docx import Document
from io import BytesIO

def parse_resume(file) -> str:
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(file.file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text

    elif filename.endswith(".docx"):
        doc = Document(BytesIO(file.file.read()))
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return ""
