import logging
import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
# Set timeout to 10 minutes (600 seconds) for slow connections
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------
model_name = "google/flan-t5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
kw_model = KeyBERT(model=embed_model)

# --------------------------------------------------
# Text Cleaning & Deduplication
# --------------------------------------------------
def clean_sentences(text):
    sentences = re.split(r"[.!?]", text)
    seen = set()
    cleaned = []
    for s in sentences:
        s_clean = re.sub(r"\s+", " ", s).strip()
        # Only keep unique sentences to prevent 'internship and internship' [cite: 62, 64]
        if len(s_clean) > 20 and s_clean.lower() not in seen:
            cleaned.append(s_clean)
            seen.add(s_clean.lower())
    return cleaned
# --------------------------------------------------
# Semantic Logic (Extractive)
# --------------------------------------------------
def extract_important_text(text, keep_ratio=0.6):
    """Ranks and keeps the most relevant sentences."""
    sentences = clean_sentences(text)
    if not sentences: return text

    embeddings = embed_model.encode(sentences)
    doc_embedding = embeddings.mean(axis=0)
    scores = cosine_similarity([doc_embedding], embeddings)[0]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    top_n = max(3, int(len(sentences) * keep_ratio))
    
    return ". ".join([s for s, _ in ranked[:top_n]])

def extract_key_points(text, top_n=5):
    """
    Generates unique bullet points and prevents repeating the same 
    project multiple times.
    """
    sentences = clean_sentences(text)
    if not sentences: return ""

    embeddings = embed_model.encode(sentences)
    doc_embedding = embeddings.mean(axis=0)
    scores = cosine_similarity([doc_embedding], embeddings)[0]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    
    unique_bullets = []
    seen_content = set()

    for s, score in ranked:
        # Check first 25 chars to ensure bullets cover different topics
        prefix = s[:25].lower()
        if prefix not in seen_content:
            unique_bullets.append(f"- {s}")
            seen_content.add(prefix)
        
        if len(unique_bullets) >= top_n:
            break

    return "\n".join(unique_bullets)

# --------------------------------------------------
# Chunking Logic
# --------------------------------------------------
def split_into_semantic_chunks(text, max_words=350):
    """Groups sentences to avoid context loss during processing."""
    sentences = clean_sentences(text)
    chunks, current_chunk, current_count = [], [], 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count <= max_words:
            current_chunk.append(sentence)
            current_count += word_count
        else:
            chunks.append(". ".join(current_chunk))
            current_chunk, current_count = [sentence], word_count
    
    if current_chunk: 
        chunks.append(". ".join(current_chunk))
    return chunks[:8]

# --------------------------------------------------
# AI Generation (Abstractive)
# --------------------------------------------------
def generate(prompt, is_final=False):
    """Generates text with low temperature to stop hallucinations."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150 if is_final else 100,
            min_length=40,
            do_sample=True,
            top_p=0.92,
            temperature=0.5, # Low temp for factual grounding
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def generate_ai_insights(text):
    """Coordinates extraction with strict grounding constraints."""
    start = time.time()
    
    important_text = extract_important_text(text)
    chunks = split_into_semantic_chunks(important_text)
    
    chunk_summaries = []
    for chunk in chunks:
        # Prefix forcing for objective tone
        prompt = f"Facts: {chunk}\n\nObjective Summary: This document details"
        chunk_summaries.append("This document details " + generate(prompt))

    combined_text = " ".join(chunk_summaries)
    
    # Final Synthesis
    final_prompt = f"""
    Context: {combined_text}
    
    Task: Write a factual 3-sentence summary of this person's professional background.
    Rule: Start with "This document outlines". Do not use fragments like "JS, MDX".
    
    Summary:
    This document outlines"""

    final_summary = "This document outlines " + generate(final_prompt, is_final=True)
    # Final forced prefix
    final_summary = "The provided document confirms that " + generate(final_prompt, is_final=True)

    bullets = extract_key_points(important_text)
    keywords_raw = kw_model.extract_keywords(important_text, top_n=6)
    keywords = ", ".join([k[0] for k in keywords_raw])

    logger.info(f"Analysis complete in {time.time()-start:.2f}s")

    return {
        "summary": final_summary,
        "bullets": bullets,
        "keywords": keywords
    }