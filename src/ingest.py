import os
import json
import re
import fitz

PAPERS_DIR = "data/papers"
OUTPUT_PATH = "data/chunks.json"

CHUNK_WORDS = 200
OVERLAP_WORDS = 40


# -------------------------------------------------
# CLEAN TEXT (PRESERVE NEWLINES INITIALLY)
# -------------------------------------------------
def clean_text(text):
    # Normalize spacing but KEEP line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


# -------------------------------------------------
# WORD-BASED CHUNKING
# -------------------------------------------------
def chunk_by_words(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if len(chunk_words) > 50:
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


# -------------------------------------------------
# MAIN PROCESSING
# -------------------------------------------------
def process_papers():
    all_chunks = []
    chunk_id = 0

    for filename in os.listdir(PAPERS_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PAPERS_DIR, filename)
        paper_title = filename.replace(".pdf", "")

        doc = fitz.open(pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text() + "\n"

        print("RAW LENGTH:", len(full_text))

        # Clean spacing but keep structure
        full_text = clean_text(full_text)
        print("AFTER CLEAN LENGTH:", len(full_text))

        # -------------------------------------------------
        # SAFE REFERENCE REMOVAL
        # -------------------------------------------------
        ref_match = re.search(r"\nReferences\s*\n", full_text, flags=re.IGNORECASE)

        if ref_match and ref_match.start() > len(full_text) * 0.6:
            full_text = full_text[:ref_match.start()]

        print("AFTER REF REMOVAL LENGTH:", len(full_text))

        # -------------------------------------------------
        # REMOVE FIGURE CAPTIONS (SAFE VERSION)
        # -------------------------------------------------
        full_text = re.sub(
            r"\nFigure\s+\d+.*",
            "",
            full_text,
            flags=re.IGNORECASE
        )

        # -------------------------------------------------
        # REMOVE INLINE CITATIONS [123]
        # -------------------------------------------------
        full_text = re.sub(r"\[\d+\]", "", full_text)

        # -------------------------------------------------
        # NOW FLATTEN TEXT FOR CHUNKING
        # -------------------------------------------------
        full_text = re.sub(r"\n+", " ", full_text)

        print("TOTAL WORDS:", len(full_text.split()))

        # -------------------------------------------------
        # CHUNK
        # -------------------------------------------------
        chunks = chunk_by_words(full_text, CHUNK_WORDS, OVERLAP_WORDS)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": chunk,
                "paper_title": paper_title
            })
            chunk_id += 1

    return all_chunks


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    print("Processing paper...")
    chunks = process_papers()

    print(f"Total chunks: {len(chunks)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Chunks saved.")
