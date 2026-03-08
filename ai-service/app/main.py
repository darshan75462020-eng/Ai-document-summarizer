from fastapi import FastAPI, UploadFile, File
from pdf_reader import extract_text_from_pdf
from summarizer import generate_ai_insights
app = FastAPI()
@app.get("/")
def home():
    return {"message": "AI Summarizer Running"}
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file.file)
    result = generate_ai_insights(text)
    return result