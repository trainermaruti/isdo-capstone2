"""
ISDO Knowledge Base ingestion + retrieval test.

1. Reads all .md files from data/kb/
2. Splits each file into chunks at '##' headings
3. Stores chunks in a ChromaDB collection named 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence

Requires: pip install chromadb
"""

import os
import re
import chromadb

KB_DIR = "data/kb"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "How do I reset my password?",
    "VPN keeps disconnecting, how to fix?",
    "Application throwing a license error on startup",
    "How to request access to a shared drive?",
]


def load_and_chunk(kb_dir):
    """Read every .md file and split on '##' headings. Returns list of dicts."""
    chunks = []
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    md_files = [f for f in os.listdir(kb_dir) if f.lower().endswith(".md")]
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    for filename in sorted(md_files):
        path = os.path.join(kb_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split at '##' headings (keep the heading with its section content).
        # Anything before the first '##' (e.g. a '#' title) is kept as its own chunk.
        parts = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
        section_idx = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            heading_match = re.match(r"^##\s*(.+)", part)
            heading = heading_match.group(1).strip() if heading_match else "Intro"
            chunks.append(
                {
                    "id": f"{filename}::chunk{section_idx}",
                    "text": part,
                    "article": filename,
                    "heading": heading,
                }
            )
            section_idx += 1

    return chunks


def build_collection(chunks):
    """Create (or reset) the isdo_kb collection and load all chunks."""
    client = chromadb.Client()

    # Reset if it already exists, so re-runs don't duplicate data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"article": c["article"], "heading": c["heading"]} for c in chunks],
    )
    return collection


def run_queries(collection, queries):
    for query in queries:
        result = collection.query(query_texts=[query], n_results=1)

        if not result["ids"] or not result["ids"][0]:
            print(f"Query: {query}\n  No match found.\n")
            continue

        distance = result["distances"][0][0]
        metadata = result["metadatas"][0][0]
        confidence = 1 / (1 + distance)  # convert distance to a 0-1-ish confidence score

        print(f"Query: {query}")
        print(f"  Best match : {metadata['article']}  (section: {metadata['heading']})")
        print(f"  Distance   : {distance:.4f}")
        print(f"  Confidence : {confidence:.4f}")
        print()


def main():
    chunks = load_and_chunk(KB_DIR)
    print(f"Loaded {len(chunks)} chunks from {KB_DIR}\n")

    collection = build_collection(chunks)
    print(f"Stored chunks in ChromaDB collection '{COLLECTION_NAME}'\n")

    run_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()