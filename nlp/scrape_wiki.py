"""
ISL Pipeline — Wikipedia Training Data Scraper
===============================================
Automatically scrapes Wikipedia articles on topics relevant to
academic lectures and generates English-ISL gloss training pairs.

Usage:
  python scrape_wiki.py              # scrape default topics
  python scrape_wiki.py --topic "neural networks" "machine learning"
  python scrape_wiki.py --retrain    # scrape + retrain Seq2Seq after
"""

import json
import re
import time
import argparse
from pathlib import Path

import requests

BASE_DIR    = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "training_pairs.json"

# ── Topics relevant to academic lectures ──────────────────────────────────────
DEFAULT_TOPICS = [
    # AI / ML
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "speech recognition",
    "transformer model",
    "reinforcement learning",

    # CS fundamentals
    "computer science",
    "data structure",
    "algorithm",
    "operating system",
    "computer network",
    "database",
    "software engineering",

    # General academic
    "mathematics",
    "probability",
    "linear algebra",
    "calculus",
    "physics",
    "chemistry",
    "biology",
    "history",
    "economics",

    # Accessibility / ISL relevant
    "sign language",
    "Indian Sign Language",
    "deaf education",
    "accessibility",
]

# ── Wikipedia API ──────────────────────────────────────────────────────────────

def fetch_wikipedia(topic: str) -> str | None:
    """Fetch plain text summary of a Wikipedia article using the API."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "ISL-Pipeline/1.0"})
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "")
        # Try searching if direct lookup fails
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "list": "search",
            "srsearch": topic, "format": "json", "srlimit": 1
        }
        r = requests.get(search_url, params=params, timeout=10,
                        headers={"User-Agent": "ISL-Pipeline/1.0"})
        if r.status_code == 200:
            results = r.json().get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                return fetch_wikipedia(title)
    except Exception as e:
        print(f"[SCRAPER] Error fetching '{topic}': {e}")
    return None


def fetch_full_wikipedia(topic: str) -> str | None:
    """Fetch full Wikipedia article text (not just summary)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action":      "query",
        "titles":      topic,
        "prop":        "extracts",
        "explaintext": True,
        "format":      "json",
        "redirects":   1,
    }
    try:
        r = requests.get(url, params=params, timeout=15,
                        headers={"User-Agent": "ISL-Pipeline/1.0"})
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
    except Exception as e:
        print(f"[SCRAPER] Error fetching full article '{topic}': {e}")
    return None

# ── Text processing ────────────────────────────────────────────────────────────

def clean_sentence(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split article text into clean sentences."""
    # Remove Wikipedia section headers
    text = re.sub(r"==+[^=]+=+", " ", text)
    # Remove citations like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove parenthetical pronunciations
    text = re.sub(r"\([^)]{0,30}\)", "", text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if len(words) < 4 or len(words) > 25:
            continue
        alpha = sum(c.isalpha() for c in sent) / max(len(sent), 1)
        if alpha < 0.65:
            continue
        # Skip sentences that are mostly proper nouns / numbers
        result.append(sent)
    return result

# ── Pair generation ────────────────────────────────────────────────────────────

def generate_pairs_from_text(text: str, source: str) -> list[dict]:
    """Convert article text to English-ISL training pairs."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    try:
        from gloss_converter import convert_sentence
    except ImportError:
        print("[SCRAPER] ERROR: gloss_converter.py not found")
        return []

    sentences = split_sentences(text)
    pairs = []
    for sent in sentences:
        clean = clean_sentence(sent)
        if not clean:
            continue
        gloss = convert_sentence(clean)
        if not gloss or gloss.lower() == clean or len(gloss.split()) < 2:
            continue
        pairs.append({
            "english": clean,
            "gloss":   gloss,
            "source":  source,
        })
    return pairs


def save_pairs(new_pairs: list[dict]) -> int:
    """Append new pairs to training_pairs.json, returns number added."""
    existing = []
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())

    seen  = {p["english"] for p in existing}
    added = [p for p in new_pairs if p["english"] not in seen]
    combined = existing + added

    OUTPUT_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return len(added)

# ── Main ───────────────────────────────────────────────────────────────────────

def scrape(topics: list[str], full_articles: bool = True) -> int:
    """Scrape Wikipedia articles and generate training pairs."""
    total_added = 0

    for i, topic in enumerate(topics):
        print(f"[SCRAPER] ({i+1}/{len(topics)}) Fetching: {topic}…", end=" ")

        text = fetch_full_wikipedia(topic) if full_articles else fetch_wikipedia(topic)
        if not text:
            print("not found")
            continue

        pairs = generate_pairs_from_text(text, f"wikipedia:{topic}")
        added = save_pairs(pairs)
        total_added += added
        print(f"{len(pairs)} pairs generated, {added} new")

        # Be polite to Wikipedia API
        time.sleep(0.5)

    return total_added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Wikipedia for ISL training data")
    parser.add_argument("--topic",   nargs="+", help="Topics to scrape (default: built-in list)")
    parser.add_argument("--retrain", action="store_true", help="Retrain Seq2Seq after scraping")
    parser.add_argument("--summary", action="store_true", help="Use summaries only (faster, less data)")
    args = parser.parse_args()

    topics = args.topic if args.topic else DEFAULT_TOPICS
    full   = not args.summary

    print(f"[SCRAPER] Scraping {len(topics)} topics…")
    total = scrape(topics, full_articles=full)

    # Show current dataset size
    if OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text())
        print(f"\n[SCRAPER] Done! Added {total} new pairs")
        print(f"[SCRAPER] Total training pairs: {len(data)}")

        # Show sample
        import random
        print(f"\n[SCRAPER] Sample pairs:")
        print(f"{'English':<45} {'ISL Gloss'}")
        print("─" * 75)
        for p in random.sample(data, min(8, len(data))):
            print(f"{p['english'][:44]:<45} {p['gloss']}")

    if args.retrain:
        print("\n[SCRAPER] Retraining Seq2Seq model…")
        import subprocess, sys
        subprocess.run([sys.executable, str(BASE_DIR / "seq2seq_gloss.py"), "--mode", "train"])