import logging
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model_name = "sshleifer/distilbart-cnn-12-6"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using compute device: {device}")

# Load models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

kw_model = KeyBERT()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Extract Important Sentences (Speed Optimization)
# --------------------------------------------------
def extract_important_text(text, keep_ratio=0.35):

    sentences = [
        re.sub(r"[^a-zA-Z0-9 ,.-]", "", s).strip()
        for s in text.split(".")
        if len(s.strip()) > 20
    ]

    if len(sentences) == 0:
        return text

    embeddings = embed_model.encode(sentences)

    doc_embedding = embeddings.mean(axis=0)

    scores = cosine_similarity([doc_embedding], embeddings)[0]

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_n = int(len(sentences) * keep_ratio)

    important_sentences = [s for s, _ in ranked[:top_n]]

    logger.info(f"Reduced text from {len(sentences)} to {top_n} important sentences.")

    return ". ".join(important_sentences)


# --------------------------------------------------
# Key Bullet Points
# --------------------------------------------------
def extract_key_points(text, top_n=5):

    sentences = [
        re.sub(r"[^a-zA-Z0-9 ,.-]", "", s).strip()
        for s in text.split(".")
        if len(s.strip()) > 20
    ]

    if len(sentences) == 0:
        return ""

    embeddings = embed_model.encode(sentences)

    doc_embedding = embeddings.mean(axis=0)

    scores = cosine_similarity([doc_embedding], embeddings)[0]

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    bullets = "\n".join([f"- {s}" for s, _ in ranked[:top_n]])

    return bullets


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------
def split_text(text, chunk_size=600):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Main AI Pipeline
# --------------------------------------------------
def generate_ai_insights(text):

    start_time = time.time()

    # -------- STEP 1: Reduce Text (10x speed trick) --------
    logger.info("Extracting important sentences before summarization...")
    text = extract_important_text(text)

    # -------- Generator Function --------
    def generate(prompt, max_len=150, min_len=40):

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                repetition_penalty=2.0,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------- STEP 2: Chunking --------
    chunks = split_text(text, chunk_size=600)
    logger.info(f"Created {len(chunks)} text chunks for summarization.")

    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):

        logger.info(f"Summarizing chunk {i}/{len(chunks)}...")

        try:
            summary_text = generate(chunk)
            chunk_summaries.append(summary_text)

        except Exception as e:
            logger.error(f"Error summarizing chunk {i}: {str(e)}")

    combined_summary = " ".join(chunk_summaries)

    # -------- STEP 3: Hierarchical Summarization --------
    logger.info("Running hierarchical summarization...")

    try:
        final_summary = generate(combined_summary, max_len=200, min_len=50)

    except Exception as e:
        logger.error(f"Error in hierarchical summarization: {str(e)}")
        final_summary = combined_summary

    # -------- STEP 4: Bullet Points --------
    bullets = extract_key_points(text)

    # -------- STEP 5: Keywords --------
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )

    keywords = ", ".join([kw[0] for kw in keywords])

    end_time = time.time()
    logger.info(f"Total summarization time: {end_time - start_time:.2f} seconds.")

    return {
        "summary": final_summary,
        "bullets": bullets,
        "keywords": keywords
    }