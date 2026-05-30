"""
app.py — Flask backend for the ISL Avatar pipeline.

Replaces serve.py. Serves the frontend AND processes video uploads
end-to-end: video → audio extraction → Whisper ASR → ISL gloss → JSON.

Usage:
    conda activate base          # or whichever env has whisper + flask
    python app.py                # runs on http://localhost:8000
"""

import re
import subprocess
import tempfile
from pathlib import Path

import whisper
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).parent

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# ── Whisper model — loaded once at startup so requests are fast ──
print("[APP] Loading Whisper model (base)…")
_model = whisper.load_model("base")
print("[APP] Whisper ready.\n")


# ═══════════════════════════════════════════════════════════════
# ASR helpers  (mirrors asr/asr_isl.py)
# ═══════════════════════════════════════════════════════════════

_FILLERS = {
    "uh", "um", "uh-huh", "hmm", "hm", "ah", "er", "err",
    "like", "you know", "i mean", "right", "okay", "ok",
    "so", "basically", "actually", "literally",
}


def _clean(text: str) -> str:
    text = text.lower().strip()
    pattern = r"\b(" + "|".join(re.escape(f) for f in _FILLERS) + r")\b[,.]?\s*"
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text.rstrip(".,;:!?").strip()


def _seg_to_record(seg: dict) -> dict:
    raw   = seg["text"].strip()
    clean = _clean(raw)
    return {
        "id":         seg["id"],
        "start":      round(seg["start"], 2),
        "end":        round(seg["end"],   2),
        "raw":        raw,
        "clean":      clean,
        "word_count": len(clean.split()),
    }


# ═══════════════════════════════════════════════════════════════
# Gloss helpers  (mirrors nlp/gloss_converter.py)
# ═══════════════════════════════════════════════════════════════

_DROP = {
    "am", "is", "are", "was", "were", "be",
    "the", "a", "an", "do", "does", "did", "to", "very",
}
_NEG  = {"not", "never", "no", "don't", "doesn't", "didn't", "cannot", "can't"}
_WH   = {"what", "where", "when", "why", "who", "which", "how"}
_PRON = {
    "i": "I", "me": "I", "my": "MY", "mine": "MY",
    "you": "YOU", "your": "YOUR",
    "he": "HE",  "him": "HE",  "his": "HIS",
    "she": "SHE", "her": "HER",
    "we": "WE",  "us": "WE",  "our": "OUR",
    "they": "THEY", "them": "THEY", "their": "THEIR",
    "it": "IT",  "its": "ITS",
}
_VERB = {
    "going": "GO", "went": "GO", "goes": "GO",
    "coming": "COME", "came": "COME", "comes": "COME",
    "explaining": "EXPLAIN", "explained": "EXPLAIN",
    "understanding": "UNDERSTAND", "understood": "UNDERSTAND",
    "learning": "LEARN", "learned": "LEARN",
    "teaching": "TEACH", "taught": "TEACH",
    "looking": "LOOK", "looked": "LOOK",
    "talking": "TALK", "talked": "TALK",
    "saying": "SAY", "said": "SAY",
    "doing": "DO", "done": "DO",
    "having": "HAVE", "had": "HAVE",
    "using": "USE", "used": "USE",
    "called": "CALL", "calling": "CALL",
    "known": "KNOW", "knowing": "KNOW",
    "shown": "SHOW", "showing": "SHOW",
    "working": "WORK", "worked": "WORK",
    "running": "RUN", "ran": "RUN",
    "writing": "WRITE", "wrote": "WRITE",
    "reading": "READ",
    "helping": "HELP", "helped": "HELP",
    "starting": "START", "started": "START",
    "stopping": "STOP", "stopped": "STOP",
}


def _to_token(word: str):
    if word in _DROP:   return None
    if word in _NEG:    return "NOT"
    if word in _PRON:   return _PRON[word]
    if word in _VERB:   return _VERB[word]
    return word.upper()


def _convert(text: str) -> str:
    words = [w.strip("'") for w in re.sub(r"[^\w\s']", "", text.lower()).split() if w.strip("'")]
    if not words:
        return ""

    wh_word     = words[0].upper() if words[0] in _WH else None
    not_pending = False
    tokens      = []

    for w in words:
        if w in _NEG:
            not_pending = True
            continue
        if w in _WH and wh_word:
            continue
        t = _to_token(w)
        if t:
            tokens.append(t)

    # OSV reorder (skip for WH questions)
    if not wh_word and tokens and tokens[0] in set(_PRON.values()) and len(tokens) >= 3:
        tokens = tokens[2:] + [tokens[0], tokens[1]]

    if not_pending:
        tokens.append("NOT")
    if wh_word:
        tokens.append(wh_word)

    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR / "avatar"), "avatar_3d.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "whisper-base"})


@app.route("/api/process-video", methods=["POST"])
def process_video():
    """
    Accepts a video file upload, runs the full pipeline, and returns
    gloss_output.json as a JSON array.

    Steps:
      1. Save uploaded video to a temp directory
      2. Extract audio track to 16 kHz mono WAV using ffmpeg
      3. Transcribe with Whisper (base model)
      4. Convert each segment to ISL gloss
      5. Return the structured JSON array
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files["video"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp        = Path(tmpdir)
        video_path = tmp / "input.mp4"
        audio_path = tmp / "audio.wav"

        video_file.save(str(video_path))

        # ── Step 1: extract audio ──────────────────────────────
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-ar", "16000", "-ac", "1",   # 16 kHz mono — what Whisper expects
                "-y", str(audio_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            return jsonify({
                "error": "ffmpeg failed: " + result.stderr.decode(errors="replace")
            }), 500

        # ── Step 2: transcribe ─────────────────────────────────
        transcription = _model.transcribe(str(audio_path), language="en", fp16=False)

        # ── Step 3: build gloss records ────────────────────────
        output = []
        for seg in transcription["segments"]:
            rec   = _seg_to_record(seg)
            gloss = _convert(rec["clean"])
            output.append({
                "id":         rec["id"],
                "start":      rec["start"],
                "end":        rec["end"],
                "clean":      rec["clean"],
                "gloss":      gloss,
                "word_count": rec["word_count"],
            })

        return jsonify(output)


if __name__ == "__main__":
    import sys
    print(f"[APP] Python  → {sys.executable}")
    print("[APP] Server  → http://localhost:8000")
    print("[APP] Open that URL in your browser (not the HTML file directly).")
    app.run(host="0.0.0.0", port=8000, debug=False)
