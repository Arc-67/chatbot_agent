# ingest_simple.py
import os
from dotenv import load_dotenv
import json
import re
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ------------- Config -------------
load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "mind-hive-test") #

if not OPENAI_KEY or not PINECONE_API_KEY:
    raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in environment.")

# ------------- Helpers -------------
def normalize_price(raw_price: str) -> str:
    if not raw_price:
        return ""
    s = (raw_price or "").replace(",", "")
    m = re.search(r"(RM|USD|SGD|EUR|\$)?\s*[\d]+(?:\.\d+)?", s, re.IGNORECASE)
    return m.group(0).strip() if m else raw_price.strip()

def clean_variants(variants):
    seen = set()
    out = []
    for v in variants or []:
        lbl = (v.get("label") or "").strip()
        if not lbl:
            continue
        if lbl in seen:
            continue
        seen.add(lbl)
        out.append(lbl)
    return out

# example output
# variants = [
#   {"id": "1", "label": " Red "},
#   {"id": "2", "label": "Blue"},
#   {"id": "3", "label": ""},           # skipped
#   {"id": "4", "label": "Red"},        # duplicate -> skipped
#   {"id": "5"}                         # no label -> skipped
# ]
# clean_variants(variants)
# -> ["Red", "Blue"]

def make_document_text(product: Dict) -> str:
    title = (product.get("title") or "").strip()
    price = normalize_price(product.get("price", ""))
    variants = clean_variants(product.get("variants", []))
    variants_text = "; ".join(variants) if variants else ""
    desc = (product.get("description_text") or "").strip() or (product.get("description_html") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if price:
        parts.append(f"Price: {price}")
    if variants_text:
        parts.append(f"Variants: {variants_text}")
    if desc:
        parts.append(desc)
    final = "\n\n".join(parts)
    if not final:
        final = title or price or product.get("product_id", "")
    return final

# ------------- Main -------------
def main(jsonl_path: str = "products.jsonl"):
    # load products
    texts: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping invalid JSON line {line_no}: {e}")
                continue

            pid = product.get("product_id") or product.get("url") or f"prod_{line_no}"
            text = make_document_text(product).strip()
            if not text:
                print(f"[WARN] Skipping {pid} because composed text is empty")
                continue

            # small metadata for display & identification
            metadata = {
                "product_id": pid,
                "title": product.get("title"),
                "url": product.get("url"),
                "price": normalize_price(product.get("price","")),
                "variants": clean_variants(product.get("variants", [])),
                "last_crawled": product.get("last_crawled"),
                "source": product.get("source", "zus_drinkware"),
                "snippet": text[:400]
            }

            texts.append(text)
            metadatas.append(metadata)
            ids.append(pid)

    if not texts:
        print("[INFO] No products found to index.")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")  # default model; set model param if needed
    
    print(f"[INFO] Indexing {len(texts)} documents to Pinecone index '{INDEX_NAME}'...")
    # print(texts[0])
    # print(metadatas[0])

    # Connect to Pinecone vectorstore using the Index name
    store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, text_key="text")

    # add the documents (internally will compute embeddings and upsert), add_texts is efficient for small collections
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    print("[INFO] Done indexing.")

if __name__ == "__main__":
    main()
