import fitz
import re

def clean_pdf_text(text):
    """
    Refined cleaning logic to preserve resume structure and technical terms.
    Ensures contact info and list items remain intact.
    """
    # 1. Remove (cid:x) artifacts common in specific PDF encodings
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"cid\d+", "", text)

    # 2. Rejoin words split by line-break hyphens (e.g., "de- velopment" -> "development")
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # 3. Handle list markers (preserving bullets for the summarizer to recognize lists) [cite: 48, 50]
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]", " ", text)

    # 4. Normalize spacing while preserving paragraph intent
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text) # Keep single newlines for structure
    text = re.sub(r"\s+", " ", text)

    # 5. Remove full-caps headers only if they are long (>6 chars) 
    # to protect short acronyms like AI, SQL, NLP, API 
    text = re.sub(r"\b[A-Z]{7,}\b", "", text)

    # 6. Final ASCII check to remove encoding noise but keep standard characters
    text = re.sub(r'[^\x00-\x7f]', r'', text)

    return text.strip()


def extract_text_from_pdf(file):
    """
    Extracts text using 'blocks' to maintain the natural reading order 
    of resumes and academic papers.
    """
    text = ""

    try:
        # Handle file-like objects or direct paths
        if hasattr(file, "read"):
            file.seek(0)
            file_bytes = file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            doc = fitz.open(file)

        for page in doc:
            # Using 'blocks' helps preserve table and column relationships in resumes 
            blocks = page.get_text("blocks")
            
            # Sort blocks by vertical (y) position then horizontal (x) position
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    text += block_text + " "

        doc.close()

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

    # Clean the extracted text using the updated professional rules
    text = clean_pdf_text(text)

    return text