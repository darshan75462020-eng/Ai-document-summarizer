from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_reader import extract_text_from_pdf
from summarizer import generate_ai_insights
import logging

# Setup logging to match your AI service logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Summarizer API")

# Add CORS Middleware so your Streamlit frontend can talk to this API 
# even if they run on different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Summarizer Backend is Running",
        "version": "1.1"
    }

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    """
    Receives a PDF, extracts text, and generates AI insights.
    """
    # 1. Validate File Type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        logger.info(f"Received file: {file.filename}")

        # 2. Extract Text
        # Using await file.read() is safer for larger uploads in FastAPI
        content = await file.read()
        
        # We pass the bytes directly or wrap in a BytesIO for fitz
        import io
        text = extract_text_from_pdf(io.BytesIO(content))

        if not text or len(text.strip()) < 50:
            logger.warning("Extraction failed or text too short.")
            raise HTTPException(status_code=422, detail="Could not extract sufficient text from PDF.")

        # 3. Generate Insights
        logger.info("Sending text to AI Summarizer...")
        result = generate_ai_insights(text)

        return result

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    finally:
        # Ensure the file stream is closed
        await file.close()