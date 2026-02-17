import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------
# Config
# ---------------------------
PAPERS_DIR = "data/papers"
OUTPUT_PATH = "data/chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700        # tokens
CHUNK_OVERLAP = 100     # tokens

# ---------------------------
# Initialize Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_text_from_pdf(pdf_path):
    """Extract text page by page with metadata."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text = text.replace("\n", " ").strip()

        if len(text) > 50:  # ignore very small fragments
            pages.append({
                "text": text,
                "page_number": page_num + 1
            })

    return pages


def chunk_text(text, chunk_size=700, overlap=100):
    """Token-aware chunking."""
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def process_papers():
    all_chunks = []
    chunk_id = 0

    for filename in tqdm(os.listdir(PAPERS_DIR)):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PAPERS_DIR, filename)
        paper_title = filename.replace(".pdf", "")

        pages = extract_text_from_pdf(pdf_path)

        for page in pages:
            chunks = chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in chunks:
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk,
                    "paper_title": paper_title,
                    "page_number": page["page_number"]
                })
                chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    print("Processing papers...")
    chunks = process_papers()

    print(f"\nTotal chunks created: {len(chunks)}\n")

    if len(chunks) > 0:
        print("Sample chunk:\n")
        print(json.dumps(chunks[0], indent=2)[:1000])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nChunks saved to {OUTPUT_PATH}")
