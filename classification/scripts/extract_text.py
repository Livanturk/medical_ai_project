import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

pdf_text = extract_text_from_pdf("classification/data/PE-Brain-tumors_UCNI.pdf")

with open("classification/data/brain_tumor_text.txt", "w", encoding="utf-8") as f:
    f.write(pdf_text)

print("PDF text extracted and saved!")
