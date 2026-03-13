import fitz
import re

def extract_text_from_pdf(file):
    text = ""
    try:
        if hasattr(file, 'read'):
            file.seek(0)
            file_bytes = file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            doc = fitz.open(file)
            
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text
                
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        
    # Clean up (cid:104) or cid104 artifacts that cause ML hallucinations
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"cid\d+", "", text)
    
    return text