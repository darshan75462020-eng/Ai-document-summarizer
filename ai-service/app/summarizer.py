from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

model_name = "google/flan-t5-base"

# Load summarization model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load keyword model
kw_model = KeyBERT()

# Load embedding model for key points
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def split_text(text, chunk_size=800):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def generate_ai_insights(text):

    def generate(prompt):

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_beams=5,
                repetition_penalty=2.5,
                early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------- CHUNK SUMMARIZATION --------

    chunks = split_text(text)

    chunk_summaries = []

    for chunk in chunks:

        prompt = f"""
Summarize the following document section in 2 sentences.

Document:
{chunk}

Summary:
"""

        chunk_summary = generate(prompt)
        chunk_summaries.append(chunk_summary)

    combined_summary = " ".join(chunk_summaries)

    final_prompt = f"""
Combine the following summaries into one clear 4 sentence summary.

Summaries:
{combined_summary}

Final Summary:
"""

    summary = generate(final_prompt)

    # -------- KEY POINTS --------

    bullets = extract_key_points(text)

    # -------- KEYWORDS --------

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )

    keywords = ", ".join([kw[0] for kw in keywords])

    return {
        "summary": summary,
        "bullets": bullets,
        "keywords": keywords
    }